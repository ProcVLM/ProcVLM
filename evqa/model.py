import gc
import torch
import torch.nn as nn
import re
import json
import queue
import threading
import contextlib
import torch.multiprocessing as mp
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union

from PIL import Image
from qwen_vl_utils import process_vision_info
from qwen_vl_utils.vision_process import smart_resize, SPATIAL_MERGE_SIZE
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)
from transformers.modeling_outputs import CausalLMOutputWithPast


# ================================================
# Model Definition
# ================================================

class AttentionPooling(nn.Module):
    """ Simple Attention Pooling Module
    
    Implemented as a two-layer MLP that produces attention scores over the 
    sequence, followed by a weighted sum to get a single pooled vector. """
    
    def __init__(self, hidden_size):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(self, hidden_states, attention_mask=None):
        scores = self.score(hidden_states).squeeze(-1)
        if attention_mask is not None:
            min_val = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(attention_mask == 0, min_val)

        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)
        return pooled


class ProcVLMWithValueHead(Qwen3VLForConditionalGeneration):
    """ Procedural VLM with Value Head for Embodied Reasoning and Rewarding

    Adapted from Qwen3VL base, this model adds an attention pooling layer over 
    the full sequence of hidden states and a dedicated value head MLP to predict
    a scalar.
    
    The training objective combines ordinary CE loss for language modeling with 
    an auxiliary regression loss on the predicted value head output, supervised 
    by progress labels if present in the training data. Formally:

        L = L_CE (gt_text, pred_text) + λ * L1 (pred_progress, gt_progress)

    The weight λ referenced as `value_loss_weight` is a tunable hyperparameter.
    If some samples do not contain progress labels, the value loss is masked
    out and does not contribute to the average loss for that batch.
    
    To avoid answer leakage in teacher-forcing training, we apply feature-level 
    dropout to the tail tokens, as progress labels are located at the end of the 
    training sequences, like `reasoning... <progress>84.13%</progress>`. 
    
    A fixed ratio of 50% dropout is applied to the tail hidden states, before 
    value head pooling, at the training time.

    At inference time, the model can generate text autoregressively as usual, or
    optionally run the value head to refine the generated progress estimation by
    automatically replacing the <progress>XX.XX%</progress> tag. The replacement
    is default disabled to achieve better generalization. Pass
    `enable_value_head=True` to `self.batch_infer()` to turn it on. 

    For inference or evaluation usage, refer to `is_procvlm_checkpoint()`,
    `load_procvlm()` and `batch_chat_with_value_head()` utilities below. """

    def __init__(
            self, 
            config,
            tokenizer, 
            value_loss_weight=0.2, 
            value_dropout=0, 
            value_noise_std=0.03, 
            value_loss_type="l1"
        ):
        """ Args:
        - config:
            The Qwen3VL config used to initialize the base model weights.
        - tokenizer: 
            The corresponding tokenizer for decoding progress from labels.
        - value_loss_weight: 
            λ, the weight for the auxiliary value loss in the total loss.
        - value_dropout: 
            Dropout probability for the value head MLP to prevent overfitting and 
            encourage robustness. Two dropout layers are applied with same rate.
        - value_noise_std: 
            Standard deviation of Gaussian noise injected into pooled representation
            during training for exposure bias mitigation.
        - value_loss_type: 
            Type of regression loss to use ("l1" or "smooth_l1").
        """
        super().__init__(config)
        self.tokenizer = tokenizer
        hidden_size = getattr(config, 'hidden_size', None) or config.text_config.hidden_size

        # attention pooling
        self.pooler = AttentionPooling(hidden_size)

        # value head (wider + deeper MLP for better regression capacity)
        mid_size = hidden_size * 4
        self.value_head = nn.Sequential( # 20M params for hidden_size=1536
            nn.Linear(hidden_size, mid_size),
            nn.GELU(),
            nn.LayerNorm(mid_size),
            nn.Dropout(value_dropout),
            nn.Linear(mid_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(value_dropout),
            nn.Linear(hidden_size, 1),
        )

        # hyperparameters for value loss computation
        self.value_loss_weight = value_loss_weight
        if value_loss_type not in {"l1", "smooth_l1"}:
            raise ValueError(f"Unsupported value_loss_type: {value_loss_type}. Supported types: 'l1', 'smooth_l1'.")
        self.value_loss_type = value_loss_type
        
        # Gaussian noise std injected into pooled repr during training to
        # simulate the hidden-state distribution shift between teacher-forcing
        # and autoregressive generation (exposure bias mitigation)
        self.value_noise_std = value_noise_std

        # Number of tail tokens to mask from pooler (covers <progress>XX.XX%</progress> + EOS)
        # Measured: "Therefore, ... <progress>84.13%</progress>.<|im_end|>" = 19 tokens
        # 16 masks from "is <progress>84.13%</progress>.<|im_end|>" onwards
        self._progress_tail_len = 14

        # progress regex (text-level matching after decode, avoids tokenizer boundary issues)
        self.progress_pattern = re.compile(r"<progress>(.*?)</progress>")
        self._pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        
        # getting the output embedding layer for hook
        self.lm_head = self.get_output_embeddings()
    

    # ------------------------------------------------
    # loading and saving
    # ------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        tokenizer,
        device_map="auto",
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
        value_loss_weight=0.2,
        value_dropout=0,
        value_noise_std=0.03,
        value_loss_type="l1",
        **kwargs,
    ):  
        # LoRA adapter loading path
        if _has_lora_adapter(model_path):
            from peft import PeftConfig, PeftModel
            peft_config = PeftConfig.from_pretrained(model_path)
            base_model_path = peft_config.base_model_name_or_path
            if not base_model_path:
                raise ValueError(
                    f"Cannot resolve `base_model_name_or_path` from LoRA adapter: {model_path}\n"
                    "It should be the same as the model path used in the LoRA training script.\n"
                    f"Can you manually specify it in {model_path}/adapter_config.json and try again?"
                )
            print(f"Loading base model '{base_model_path}' for LoRA adapter...")
            base_procvlm_model = cls.from_pretrained(
                model_path=base_model_path,
                tokenizer=tokenizer,
                device_map=device_map,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
                value_loss_weight=value_loss_weight,
                value_dropout=value_dropout,
                value_noise_std=value_noise_std,
                value_loss_type=value_loss_type,
                **kwargs,
            )
            print(f"Applying LoRA adapter from '{model_path}'...")
            model = PeftModel.from_pretrained(base_procvlm_model, model_path, is_trainable=False)
            head_state = _load_procvlm_head_state(model_path)
            if head_state is not None:
                print("Loading fine-tuned value head weights from adapter directory...")
                model.load_state_dict(head_state, strict=False)

            # LoRA adapter is already merged into the base_model's forward pass
            model.eval()
            return model

        # Normal loading path without LoRA adapter
        procvlm_state = _load_procvlm_state_dict(model_path)
        target_dtype = _resolve_effective_dtype(torch_dtype, attn_implementation)
        target_device = _resolve_target_device(device_map)

        if procvlm_state is not None:
            # Load directly from config and then restore procvlm state.
            base_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            if hasattr(base_config, "_attn_implementation"):
                base_config._attn_implementation = attn_implementation
            else:
                setattr(base_config, "_attn_implementation", attn_implementation)

            # Enforce parameters to be created on target_device to avoid cpu/cuda mismatch.
            with _safe_device_context(target_device):
                model = cls(
                    base_config,
                    tokenizer=tokenizer,
                    value_loss_weight=value_loss_weight,
                    value_dropout=value_dropout,
                    value_noise_std=value_noise_std,
                    value_loss_type=value_loss_type,
                )
            if target_dtype is not None:
                model = model.to(dtype=target_dtype)

            model.load_state_dict(procvlm_state, strict=False)
            # Latest format may keep ProcVLM heads in standalone procvlm_head.bin.
            head_state = _load_procvlm_head_state(model_path)
            if head_state is not None:
                model.load_state_dict(head_state, strict=False)
            model.eval()
            return model

        # Standard clean Hugging Face loading
        load_kwargs = dict(trust_remote_code=True)
        # ProcVLM adds extra heads that are absent in vanilla Qwen checkpoints.
        # Keep full-parameter initialization path to avoid meta tensor dispatch errors.
        load_kwargs["low_cpu_mem_usage"] = False
        if target_dtype is not None:
            load_kwargs["torch_dtype"] = target_dtype
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        load_kwargs["attn_implementation"] = attn_implementation

        load_kwargs.update(
            value_loss_weight=value_loss_weight,
            value_dropout=value_dropout,
            value_noise_std=value_noise_std,
            value_loss_type=value_loss_type,
        )

        load_kwargs.update(kwargs)
        # Pass tokenizer as positional model arg so it goes to ProcVLM __init__,
        # instead of being forwarded to GenerationConfig kwargs (JSON serialization issue).
        model = super().from_pretrained(model_path, tokenizer, **load_kwargs)
        # For latest format, restore standalone ProcVLM heads if present.
        head_state = _load_procvlm_head_state(model_path)
        if head_state is not None:
            model.load_state_dict(head_state, strict=False)
        model.eval()
        return model


    def save_pretrained(
        self, 
        output_dir: str, 
        state_dict: Dict[str, torch.Tensor] | None = None,
        save_value_head: bool = False,
    ):
        """ Save only in latest format.

        Always writes base Qwen-compatible weights to `pytorch_model.bin`. If `save_value_head=True`
        , writes ProcVLM heads to `procvlm_extra/procvlm_head.pt`. 
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.config.save_pretrained(output_path)
        if state_dict is None:
            state_dict = self.state_dict()

        # Always save in the latest format:
        # - main checkpoint: pure Qwen-compatible weights only
        # - ProcVLM-specific heads: optional standalone procvlm_head.bin
        base_state, head_state = _split_base_and_head_state(state_dict)
        torch.save(base_state, output_path / "pytorch_model.bin")

        if save_value_head and head_state:
            head_path = output_path / _PROCVLM_HEAD_RELATIVE_PATH
            head_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(head_state, head_path)


    def state_dict(self, *args, **kwargs):
        """Return native keys and break shared-storage aliases.

        Trainer checkpoint saving may use safetensors, which rejects multiple keys sharing the same 
        underlying storage. We keep all keys but clone repeated-storage tensors to make serialization
        robust. """
        raw_state = super().state_dict(*args, **kwargs)
        output_state: Dict[str, torch.Tensor] = {}
        for key, value in raw_state.items():
            output_state[key] = value

        storage_owner: Dict[int, str] = {}
        for key, tensor in list(output_state.items()):
            if not isinstance(tensor, torch.Tensor):
                continue
            if tensor.device.type == "meta":
                continue

            try:
                storage_ptr = tensor.untyped_storage().data_ptr()
            except Exception:
                continue

            owner = storage_owner.get(storage_ptr)
            if owner is None:
                storage_owner[storage_ptr] = key
                continue

            output_state[key] = tensor.clone()

        return output_state


    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        """ Accept both legacy wrapped keys and current native keys. """
        adapted_state: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            adapted_state[_remap_legacy_checkpoint_key(key)] = value

        return super().load_state_dict(adapted_state, strict=strict)


    # ------------------------------------------------
    # utilities
    # ------------------------------------------------
    def _ensure_value_head_device(self, device: torch.device):
        """ Keep value modules on the same device as hidden states during model-parallel inference. """
        if next(self.pooler.parameters()).device != device:
            self.pooler.to(device)
        if next(self.value_head.parameters()).device != device:
            self.value_head.to(device)


    @staticmethod
    def _get_image_pixels(
        img_source, factor: int, min_pixels: Optional[int], max_pixels: Optional[int],
    ) -> Optional[int]:
        """Return post-resize pixel count from image header (no decode). None on failure."""
        try:
            if isinstance(img_source, Image.Image):
                w, h = img_source.size
            elif isinstance(img_source, str) and not img_source.startswith(("http://", "https://", "data:")):
                path = img_source[7:] if img_source.startswith("file://") else img_source
                with Image.open(path) as im:
                    w, h = im.size
            else:
                return None  # URL / base64 — can't cheaply read header
            h_bar, w_bar = smart_resize(h, w, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)
            return h_bar * w_bar
        except Exception:
            return None

    def _chunk_batch_by_vram(
        self,
        batch_items: List[Dict[str, Any]],
        device: torch.device,
        processor,
        base_cost_mb: int,
        image_cost_mb: int,
    ) -> List[Tuple[List[int], List[Dict[str, Any]]]]:
        """Dynamically chunk batch items based on available VRAM.

        Uses max-item cost model: the batch is padded to the longest sequence, so the effective cost
        is ``max_item_cost * chunk_size``, not the sum.

        Items are sorted by estimated cost (descending) before bin-packing so that expensive items
        (many / high-res images) form their own small chunks while cheap items pack tightly together.
        Original indices are preserved in the returned tuples for correct result mapping.

        Per-image cost is proportional to its post-resize pixel count (obtained via a PIL lazy header
        read + ``smart_resize``).  ``image_cost_mb`` is treated as the cost for a reference 480x480
        image; actual costs scale linearly.  Falls back to the flat ``image_cost_mb`` for URLs,
        base64 strings, or when the header read fails. """
        if device.type != "cuda":
            return [(list(range(len(batch_items))), batch_items)]

        # Derive resize parameters from the processor (matches what the actual
        # vision pipeline will do later in the producer thread).
        ip = processor.image_processor
        factor = getattr(ip, 'patch_size', 14) * getattr(ip, 'merge_size', SPATIAL_MERGE_SIZE)
        min_px = getattr(ip, 'min_pixels', None)
        max_px = getattr(ip, 'max_pixels', None)

        # Reference pixel count: image_cost_mb is calibrated for ~480x480 images.
        ref_h, ref_w = smart_resize(480, 480, factor=factor, min_pixels=min_px, max_pixels=max_px)
        ref_pixel_count = ref_h * ref_w
        cost_per_pixel = image_cost_mb / ref_pixel_count

        # Pre-compute per-item costs.
        indexed_costs: List[Tuple[int, Dict[str, Any], float]] = []
        for idx, item in enumerate(batch_items):
            imgs = item.get("image", [])
            if isinstance(imgs, str):
                imgs = [imgs]
            img_cost = 0.0
            for img in imgs:
                pixels = self._get_image_pixels(img, factor, min_px, max_px)
                if pixels is not None:
                    img_cost += cost_per_pixel * pixels
                else:
                    img_cost += image_cost_mb
            indexed_costs.append((idx, item, base_cost_mb + img_cost))

        # Sort expensive-first: large items get their own small chunks,
        # cheap items at the tail pack tightly.
        indexed_costs.sort(key=lambda x: x[2], reverse=True)

        free_vram, _ = torch.cuda.mem_get_info(device)
        safe_free_mb = (free_vram / (1024 ** 2)) * 0.9

        chunks: List[Tuple[List[int], List[Dict[str, Any]]]] = []
        current_chunk: List[Dict[str, Any]] = []
        current_indices: List[int] = []
        current_max_cost_mb = 0.0

        for orig_idx, item, item_cost_mb in indexed_costs:
            new_max = max(current_max_cost_mb, item_cost_mb)
            new_chunk_cost = new_max * (len(current_chunk) + 1)

            if current_chunk and new_chunk_cost > safe_free_mb:
                chunks.append((current_indices, current_chunk))
                current_chunk = []
                current_indices = []
                current_max_cost_mb = 0.0
                new_max = item_cost_mb

            current_chunk.append(item)
            current_indices.append(orig_idx)
            current_max_cost_mb = new_max

        if current_chunk:
            chunks.append((current_indices, current_chunk))

        return chunks


    def _decode_and_extract_progress(self, labels):
        """ Decode labels to text and extract progress info via regex.
        Returns: (progress_mask [B], gt_values [B]) """
        decodable = labels.clone()
        decodable[decodable == -100] = self._pad_token_id
        texts = self.tokenizer.batch_decode(decodable, skip_special_tokens=False)

        mask = []
        values = []
        for text in texts:
            m = self.progress_pattern.search(text)
            if m:
                mask.append(True)
                try:
                    values.append(float(m.group(1).replace("%", "").strip()))
                except Exception:
                    values.append(0.0)
            else:
                mask.append(False)
                values.append(0.0)

        return (
            torch.tensor(mask, device=labels.device, dtype=torch.bool),
            torch.tensor(values, device=labels.device, dtype=torch.float32),
        )


    @torch.no_grad()
    def _predict_progress_values(self, generated_ids, visual_kwargs: Optional[Dict[str, torch.Tensor]] = None):
        """ Forward pass on full sequence (generated_ids = input + output) → hidden states
            → pooler → value_head → predicted progress [B]. (fp 0-100) """
        B, L = generated_ids.shape
        attn_mask = (generated_ids != self._pad_token_id).long()

        if visual_kwargs is None:
            visual_kwargs = {}

        # Temporarily replace lm_head with a lightweight stub so that
        # super().forward() skips the massive [B, L, vocab_size] logits
        # allocation.  The hook still fires, giving us the hidden states.
        captured_hs = {}

        def hook_fn(module, hook_args, output):
            captured_hs["val"] = hook_args[0]

        class _IdentityHead(torch.nn.Module):
            def forward(self, x):
                return x[:, :1, :1]  # tiny slice — result is discarded

        original_lm_head = self.lm_head
        stub = _IdentityHead()
        handle = stub.register_forward_hook(hook_fn)
        self.lm_head = stub
        try:
            captured_hs.pop("val", None)
            super().forward(
                input_ids=generated_ids,
                attention_mask=attn_mask,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
                **visual_kwargs,
            )
            full_hidden_states = captured_hs.get("val", None)
            if full_hidden_states is None:
                raise RuntimeError("Failed to capture hidden states from lm_head hook.")
        finally:
            self.lm_head = original_lm_head
            handle.remove()

        # tail progress tag will not be masked during generation,
        # as it causes no harm at inference time

        self._ensure_value_head_device(full_hidden_states.device)
        pooler_dtype = next(self.pooler.parameters()).dtype

        hs_for_pooler = full_hidden_states.to(dtype=pooler_dtype)
        attn_mask_for_pooler = attn_mask.to(device=hs_for_pooler.device)
        
        pooled = self.pooler(hs_for_pooler, attn_mask_for_pooler)
        pooled = pooled.to(self.value_head[0].weight.dtype)

        return torch.sigmoid(self.value_head(pooled).squeeze(-1)) * 100.0


    # ------------------------------------------------
    # value loss computation
    # ------------------------------------------------
    def _compute_value_loss_standard(self, hidden_states, labels, attention_mask, progress_gt=None):
        """ Non-packed: labels [B, T], hidden_states [B, T, H], attention_mask [B, T]. """
        if progress_gt is not None:
            progress_mask = (progress_gt >= 0)
            gt_progress = progress_gt
        else:
            progress_mask, gt_progress = self._decode_and_extract_progress(labels)

         # Clone hidden states for in-place modification
        hs = hidden_states.clone()

        # Apply dropout to tail tokens (to break direct access to progress GT)
        tail_len = self._progress_tail_len  # e.g., 14
        B, T, H = hs.shape
        if self.training:
            for b in range(B):
                valid = attention_mask[b].nonzero(as_tuple=True)[0]
                if len(valid) == 0:
                    continue
                last = valid[-1].item()
                start = max(0, last - tail_len + 1)
                # Tail hidden states: [L, H]
                hs_tail = hs[b, start:last + 1]
                # 50% feature-level dropout
                hs_tail = nn.functional.dropout(hs_tail, p=0.5, training=self.training)
                # Write back modified hidden states
                hs[b, start:last + 1] = hs_tail

        # Forward through pooler and value head
        self._ensure_value_head_device(hs.device)
        pooler_dtype = next(self.pooler.parameters()).dtype

        hs_for_pooler = hs.to(dtype=pooler_dtype)
        pooled = self.pooler(hs_for_pooler, attention_mask.to(device=hs_for_pooler.device))  # [B, H]
        pooled = pooled.to(self.value_head[0].weight.dtype)
        
        # inject Gaussian noise during training to bridge train/inference gap
        if self.training and self.value_noise_std > 0:
            pooled = pooled + torch.randn_like(pooled) * self.value_noise_std
            
        value_logits = self.value_head(pooled).squeeze(-1)                # [B]

        # No progress samples in this batch — return zero loss that still
        # keeps pooler/value_head in the computation graph so DeepSpeed
        # AllReduce doesn't deadlock across ranks.
        if not progress_mask.any():
            return (value_logits * 0).sum()

        # compute value loss
        value_preds = torch.sigmoid(value_logits.float() * 1.2 - 0.1).clamp(0.0, 1.0)

        gt_probs = (gt_progress / 100.0).clamp(0.0, 1.0)
        gt_probs_f32 = gt_probs.float()

        if self.value_loss_type == "l1":
            value_loss = nn.functional.l1_loss(
                value_preds, gt_probs_f32, reduction="none"
            )
        elif self.value_loss_type == "smooth_l1":
            value_loss = nn.functional.smooth_l1_loss(
                value_preds, gt_probs_f32, reduction="none", beta=0.05
            )
        else:
            raise NotImplementedError(f"Unsupported value_loss_type: {self.value_loss_type}")
        value_loss = (value_loss * progress_mask.float()).sum() / (progress_mask.sum() + 1e-8)
        return value_loss.to(value_logits.dtype)


    def _compute_value_loss_packed(self, hidden_states, labels, attention_mask, progress_gt=None):
        """ Packed: labels [1, T], hidden_states [1, T, H], attention_mask [num_seqs+1] (cumsum).
        
        Split packed sequence into sub-sequences by boundaries, form a pseudo-batch, then reuse the 
        same logic as _compute_value_loss_standard. """
        boundaries = attention_mask.tolist()
        hs_flat = hidden_states[0]    # [T, H]
        num_seqs = len(boundaries) - 1
        
        if num_seqs == 0:
            # Still run pooler/value_head to keep them in the graph for AllReduce
            dummy_pooled = self.pooler(hidden_states[:, :1, :])
            return (self.value_head(dummy_pooled) * 0).sum()

        # progress detection: use pre-extracted values when available
        if progress_gt is not None:
            progress_mask = (progress_gt >= 0)
            gt_progress = progress_gt
        else:
            # fallback: split labels into pseudo-batch and decode
            labels_flat = labels[0]
            max_len = max(int(boundaries[i+1]) - int(boundaries[i]) for i in range(num_seqs))
            padded_labels = torch.full((num_seqs, max_len), -100, device=labels.device, dtype=labels.dtype)
            for i in range(num_seqs):
                start, end = int(boundaries[i]), int(boundaries[i + 1])
                padded_labels[i, :end - start] = labels_flat[start:end]
            progress_mask, gt_progress = self._decode_and_extract_progress(padded_labels)

        # build pseudo-batch of hidden states for pooling
        seq_lengths = [int(boundaries[i+1]) - int(boundaries[i]) for i in range(num_seqs)]
        max_len = max(seq_lengths)
        device = hidden_states.device
        dtype = hidden_states.dtype
        H = hs_flat.shape[-1]

        padded_hs = torch.zeros((num_seqs, max_len, H), device=device, dtype=dtype)
        for i, length in enumerate(seq_lengths):
            start = int(boundaries[i])
            padded_hs[i, :length] = hs_flat[start:start + length]

        padded_attn_mask = torch.zeros((num_seqs, max_len), device=device, dtype=torch.long)
        for i, length in enumerate(seq_lengths):
            if length > 0:
                padded_attn_mask[i, :length] = 1

        # Apply dropout to tail tokens in each sub-sequence
        N = self._progress_tail_len
        if self.training:
            for i, length in enumerate(seq_lengths):
                if length == 0:
                    continue
                last = length - 1
                start = max(0, last - N + 1)
                hs_tail = padded_hs[i, start:last + 1]
                hs_tail = nn.functional.dropout(hs_tail, p=0.5, training=self.training)
                padded_hs[i, start:last + 1] = hs_tail

        # Forward through pooler and value head
        self._ensure_value_head_device(padded_hs.device)
        pooler_dtype = next(self.pooler.parameters()).dtype

        hs_for_pooler = padded_hs.to(dtype=pooler_dtype)
        pooled = self.pooler(hs_for_pooler, padded_attn_mask)
        pooled = pooled.to(self.value_head[0].weight.dtype)

        # inject Gaussian noise during training to bridge train/inference gap
        if self.training and self.value_noise_std > 0:
            pooled = pooled + torch.randn_like(pooled) * self.value_noise_std

        value_logits = self.value_head(pooled).squeeze(-1)

        # No progress samples — return zero loss that keeps params in graph
        if not progress_mask.any():
            return (value_logits * 0).sum()
        
        # compute value loss
        value_preds = torch.sigmoid(value_logits.float() * 1.2 - 0.1).clamp(0.0, 1.0)

        gt_probs = (gt_progress / 100.0).clamp(0.0, 1.0)
        gt_probs_f32 = gt_probs.float()

        if self.value_loss_type == "l1":
            value_loss = nn.functional.l1_loss(
                value_preds, gt_probs_f32, reduction="none"
            )
        elif self.value_loss_type == "smooth_l1":
            value_loss = nn.functional.smooth_l1_loss(
                value_preds, gt_probs_f32, reduction="none", beta=0.05
            )
        else:
            raise NotImplementedError(f"Unsupported value_loss_type: {self.value_loss_type}")
        value_loss = (value_loss * progress_mask.float()).sum() / (progress_mask.sum() + 1e-8)
        return value_loss.to(value_logits.dtype)


    # ------------------------------------------------
    # methods
    # ------------------------------------------------
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        cache_position=None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        progress_gt=None,
        **kwargs
    ):
        """ Forward pass with integrated value loss computation. """
        # Inference/generation fast path: keep native Qwen forward behavior.
        # Value-head loss is only needed during training when labels are present.
        if labels is None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

        # Trainer/PEFT may pass output_hidden_states via kwargs; avoid duplicate kwarg.
        kwargs.pop('output_hidden_states', None)

        # using hook to capture hidden states right before the lm_head, avoiding output_hidden_states=True which causes OOM
        captured_hidden = {}
        def hook_fn(module, args, output):
            # args[0] is the input to the linear layer, which is the Last Hidden State
            captured_hidden['last_hidden_state'] = args[0]
        
        # register hook on the output embedding layer
        handle = self.lm_head.register_forward_hook(hook_fn)

        try:
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                output_hidden_states=False, 
                **kwargs
            )
        finally:
            # remove hook to avoid side effects
            handle.remove()
        
        # [B, T, H]
        hidden_states = captured_hidden['last_hidden_state']
        loss = outputs.loss
        self._last_lm_loss = loss.detach().item() if loss is not None else None

        # if labels contain progress info, compute value loss
        if labels is not None:
            is_packed = (attention_mask is not None and attention_mask.dim() == 1)
            value_loss = (
                self._compute_value_loss_packed(hidden_states, labels, attention_mask, progress_gt)
                if is_packed
                else self._compute_value_loss_standard(hidden_states, labels, attention_mask, progress_gt)
            )
            if value_loss is not None:
                self._last_value_loss = value_loss.detach().item()
                loss = loss + self.value_loss_weight * value_loss
            else:
                self._last_value_loss = 0.0
        else:
            self._last_value_loss = 0.0

        return CausalLMOutputWithPast(
            loss=loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=None, # do not return hidden states to save memory
            attentions=outputs.attentions,
        )


    @torch.no_grad()
    def _infer_chunk(
        self,
        chunk_items: List[Dict[str, Any]],
        processor,
        device: torch.device,
        model_dtype: torch.dtype,
        generation_config: dict,
        enable_value_head: bool,
        inputs=None,
        per_sample_visual_kwargs=None,
    ) -> List[str]:
        """Generate and decode for a single chunk, with OOM retry via batch halving.

        When *inputs* / *per_sample_visual_kwargs* are supplied (normal pipeline
        path), they are used directly.  On an OOM retry the method re-processes
        from *chunk_items* so that pre-prepared tensors need not be kept alive.
        """
        if inputs is None:
            inputs, per_sample_visual_kwargs = _prepare_chunk_inputs(
                chunk_items, processor, need_per_sample_visual=enable_value_head,
            )

        # Move to GPU
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.is_floating_point():
                    inputs[k] = v.to(device=device, dtype=model_dtype)
                else:
                    inputs[k] = v.to(device=device)

        try:
            generated_ids = self.generate(**inputs, **generation_config)
        except torch.cuda.OutOfMemoryError:
            # inputs.clear() releases GPU tensors even though the caller still
            # holds a reference to the dict.  gc.collect() breaks any cyclic
            # refs kept alive by the exception traceback.
            inputs.clear()
            gc.collect()
            torch.cuda.empty_cache()
            if len(chunk_items) <= 1:
                raise  # single sample OOM — cannot split further
            mid = len(chunk_items) // 2
            left = self._infer_chunk(
                chunk_items[:mid], processor, device, model_dtype,
                generation_config, enable_value_head,
            )
            right = self._infer_chunk(
                chunk_items[mid:], processor, device, model_dtype,
                generation_config, enable_value_head,
            )
            return left + right

        prompt_lens = [len(ids) for ids in inputs.input_ids]
        generated_ids_trimmed = [g[l:] for g, l in zip(generated_ids, prompt_lens)]
        output_texts = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        del generated_ids_trimmed
        del inputs

        # Optional value head inference
        if enable_value_head and per_sample_visual_kwargs:
            progress_indices = [i for i, text in enumerate(output_texts) if '<progress>' in text]
            if progress_indices:
                selected_ids = generated_ids[progress_indices].contiguous()
                del generated_ids

                merged_visual_kwargs: Dict[str, torch.Tensor] = {}
                for key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
                    tensors = []
                    for i in progress_indices:
                        value = per_sample_visual_kwargs[i].get(key, None)
                        if isinstance(value, torch.Tensor):
                            if value.is_floating_point():
                                value = value.to(device=device, dtype=model_dtype)
                            else:
                                value = value.to(device=device)
                            tensors.append(value)
                    if tensors:
                        merged_visual_kwargs[key] = torch.cat(tensors, dim=0)

                pred = self._predict_progress_values(
                    selected_ids, visual_kwargs=merged_visual_kwargs,
                )
                for local_i, i in enumerate(progress_indices):
                    output_texts[i] = self.progress_pattern.sub(
                        f'<progress>{pred[local_i].item():.2f}%</progress>',
                        output_texts[i],
                    )
                del selected_ids, merged_visual_kwargs, pred
            else:
                del generated_ids
        else:
            del generated_ids

        del per_sample_visual_kwargs
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return output_texts

    @torch.no_grad()
    def batch_infer(
        self,
        batch_items: List[Dict[str, Any]],
        processor,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        enable_value_head: bool = False,
        base_cost_mb: Optional[int] = 200,
        image_cost_mb: Optional[int] = 150,
        show_progress: bool = True,
        **generate_kwargs,
    ) -> List[str]:
        """ Args:
        - enble_value_head:
            Whether to run the value head and replace <progress>XX.XX%</progress> tags in the generated
            text with the predicted values. This is default disabled to leverage the VLM's powerful
            generalization ability.
        - base_cost_mb, image_cost_mb:
            Parameters for resolution-aware dynamic batching. ``image_cost_mb`` is the reference VRAM
            cost for a ~480x480 image; actual per-image costs scale linearly with the post-resize pixel
            count (obtained via a PIL lazy header read + ``smart_resize``). Falls back to the flat
            ``image_cost_mb`` for URLs, base64 strings, or unreadable files.
        """
        if not batch_items:
            return []

        if processor.tokenizer.padding_side != "left":
            processor.tokenizer.padding_side = "left"

        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype

        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation_config["temperature"] = temperature
            generation_config["top_p"] = generate_kwargs.pop("top_p", 0.9)
        generation_config.update(generate_kwargs)

        # auto chunking by resolution-aware VRAM cost estimation
        chunks = self._chunk_batch_by_vram(batch_items, device, processor, base_cost_mb, image_cost_mb)

        # set maxsize small to prevents RAM OOM while perfectly hiding CPU processing latency
        data_queue = queue.Queue(maxsize=2)

        def producer():
            for chunk_indices, chunk_items in chunks:
                try:
                    inputs, per_sample_visual_kwargs = _prepare_chunk_inputs(
                        chunk_items, processor, need_per_sample_visual=enable_value_head,
                    )
                    data_queue.put((chunk_indices, chunk_items, inputs, per_sample_visual_kwargs, None))
                except Exception as e:
                    data_queue.put((None, None, None, None, e))
                    break
            data_queue.put((None, None, None, None, None))  # EOF sentinel

        producer_thread = threading.Thread(target=producer)
        producer_thread.start()

        final_results = [""] * len(batch_items)

        if show_progress:
            total = len(batch_items)
            pbar = tqdm(total=total, desc="Inference with Value Head" if enable_value_head else "Inference", unit="item")

        while True:
            chunk_indices, chunk_items, inputs, per_sample_visual_kwargs, exc = data_queue.get()
            if exc is not None:
                raise exc
            if chunk_indices is None:
                break

            output_texts = self._infer_chunk(
                chunk_items, processor, device, model_dtype,
                generation_config, enable_value_head,
                inputs=inputs, per_sample_visual_kwargs=per_sample_visual_kwargs,
            )

            for local_idx, global_idx in enumerate(chunk_indices):
                final_results[global_idx] = output_texts[local_idx]

            if show_progress:
                pbar.update(len(chunk_indices))

        producer_thread.join()
        if show_progress:
            pbar.close()
        return final_results


# ===============================================
# Utilities for ProcVLM Checkpoint Detection and Loading
# ===============================================

def is_procvlm_checkpoint(model_path: str) -> bool:
    """ Heuristic checks to determine if a given model path contains a ProcVLM
        checkpoint. Checks for:
    1) Presence of "procvlm_model.bin" (single-file checkpoint)
    2) Presence of both "adapter_config.json" and "procvlm_head.bin" (for LoRA
        adapter checkpoint)
    3) For standard Hugging Face checkpoints, checks if the state dict contains
        keys indicative of ProcVLM architecture. """
    path = Path(model_path)
    if not path.exists() or not path.is_dir():
        return False

    if (path / "procvlm_model.bin").exists():
        return True

    # Latest format may store ProcVLM heads independently from base Qwen weights.
    if any((path / candidate).exists() for candidate in _PROCVLM_HEAD_CANDIDATES):
        return True

    if (path / "adapter_config.json").exists() and any(
        (path / candidate).exists() for candidate in _PROCVLM_HEAD_CANDIDATES
    ):
        return True

    index_file = path / "pytorch_model.bin.index.json"
    if index_file.exists():
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})
            if isinstance(weight_map, dict):
                return any(
                    key.startswith("base_model.")
                    or key.startswith("pooler.")
                    or key.startswith("value_head.")
                    for key in weight_map.keys()
                )
        except Exception:
            return False

    single_bin = path / "pytorch_model.bin"
    if single_bin.exists():
        try:
            state_dict = torch.load(single_bin, map_location="cpu")
            return _is_procvlm_state_dict(state_dict)
        except Exception:
            return False

    return False


def load_procvlm(
    model_path: str,
    device_map: str = "auto",
    torch_dtype: str | torch.dtype = "auto",
):
    """ Load ProcVLM checkpoint with caching to speed up repeated loads. Caches based
        on (model_path, device_map, torch_dtype).
    Args:
    - model_path: Path to the ProcVLM checkpoint directory.
    - device_map: Passed to model loading for device placement. Can be "auto" or
        specific device string.
    - torch_dtype: Passed to model loading for dtype. Can be "auto" or specific dtype
        string. """
    cache_key = (model_path, str(device_map), str(torch_dtype))
    # Fast path: return cached model immediately.
    if cache_key in _PROCVLM_CACHE:
        return _PROCVLM_CACHE[cache_key]

    # Ensure only one thread loads the same cache_key at a time.
    with _PROCVLM_INFLIGHT_LOCKS_GUARD:
        key_lock = _PROCVLM_INFLIGHT_LOCKS.get(cache_key)
        if key_lock is None:
            key_lock = threading.Lock()
            _PROCVLM_INFLIGHT_LOCKS[cache_key] = key_lock

    with key_lock:
        # Re-check after acquiring lock in case another thread populated cache.
        if cache_key in _PROCVLM_CACHE:
            return _PROCVLM_CACHE[cache_key]

        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
        )
        model = ProcVLMWithValueHead.from_pretrained(
            model_path,
            tokenizer=processor.tokenizer,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )

        with _PROCVLM_CACHE_LOCK:
            _PROCVLM_CACHE[cache_key] = (model, processor)
            return _PROCVLM_CACHE[cache_key]


def batch_chat_with_vllm(
    batch_items: List[Dict[str, Any]],
    model_path: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    tp: int = 1,
    sampling_kwargs: Optional[Dict[str, Any]] = None,
    engine_kwargs: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """ High-level utility for batch inference with ProcVLM checkpoints via vLLM
        backend. Args:
    
    - batch_items: 
        List of input items, each containing "conversations" and optional "image"/
        "video" keys.
    - model_path: 
        Path to the ProcVLM checkpoint directory.
    - max_new_tokens, temperature: 
        Generation parameters.
    - tp: 
        Tensor parallelism degree for vLLM. Not recommended for 2B models.
    - sampling_kwargs:
        Additional sampling parameters to forward to vLLM's generate() (e.g. "top_p"
        , "top_k").
    - engine_kwargs:
        Additional parameters for vLLM engine creation (e.g. "dtype", "gpu_id").
        Use "process_workers" to control number of parallel CPU workers for 
        preprocessing (default 16), and "pipeline_chunk_size" to control how many 
        items each worker processes per batch (default 8).
        
    Returns: List of generated text outputs corresponding to each input item. """
    from core.models.qwenvl import run_batch_chat_vllm
    if not batch_items:
        return []
    show_progress = True

    # Reserve wrapper-only options; do not forward them into vLLM engine kwargs.
    runtime_engine_kwargs = dict(engine_kwargs or {})
    pipeline_chunk_size = int(runtime_engine_kwargs.pop("pipeline_chunk_size", 8) or 8)
    pipeline_chunk_size = max(1, pipeline_chunk_size)
    process_workers = int(runtime_engine_kwargs.pop("process_workers", 16) or 16)
    if "dtype" not in runtime_engine_kwargs:
        runtime_engine_kwargs["dtype"] = "bfloat16" # default to bf16

    total = len(batch_items)
    ranges = [(s, min(s + pipeline_chunk_size, total)) for s in range(0, total, pipeline_chunk_size)]
    results: List[str] = [""] * total

    # Pipeline stages:
    # - Producer thread: process_batch_chat_vllm on next chunk(s)
    # - Main thread: run_batch_chat_vllm on current processed chunk
    prep_queue: queue.Queue = queue.Queue(maxsize=10)

    def _producer():
        try:
            if process_workers == 1 or len(ranges) <= 1:
                for s, e in ranges:
                    args = (s, e, batch_items[s:e], model_path, max_new_tokens, temperature, sampling_kwargs)
                    _, _, processed = _global_process_worker(args)
                    prep_queue.put((s, e, processed, None))
            else:
                worker_args = [
                    (s, e, batch_items[s:e], model_path, max_new_tokens, temperature, sampling_kwargs)
                    for s, e in ranges
                ]
                
                ctx = mp.get_context('spawn')
                with ProcessPoolExecutor(max_workers=process_workers, mp_context=ctx) as process_executor:
                    for s, e, processed in process_executor.map(_global_process_worker, worker_args):
                        prep_queue.put((s, e, processed, None))
        except Exception as exc:
            prep_queue.put((None, None, None, exc))
        finally:
            prep_queue.put((None, None, None, None))

    producer_thread = threading.Thread(target=_producer, daemon=True)
    producer_thread.start()

    if show_progress:
        pbar = tqdm(total=total, desc="vLLM Inference", unit="item")
    while True:
        s, e, processed_batch, exc = prep_queue.get()
        if exc is not None:
            raise exc
        if s is None:
            break

        preds = run_batch_chat_vllm(
            llm_inputs_batch=processed_batch,
            model_path=model_path,
            tp=tp,
            disable_async=False,
            engine_kwargs=runtime_engine_kwargs,
        )
        results[s:e] = preds
        if show_progress:
            pbar.update(e - s)
    if show_progress:
        pbar.close()

    producer_thread.join()
    return results

    
def batch_chat_with_value_head(
    batch_items: List[Dict[str, Any]],
    model_path: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    device_map: str = "auto",
    torch_dtype: str | torch.dtype = "auto",
    dp: int = 1,
    **kwargs,
) -> List[str]:
    """ High-level utility for batch inference with ProcVLM checkpoints, leveraging
        dynamic batching and multi-worker execution.

    [NOTICE] If you are encountering OOM issues during inference, consider increasing
        the base_cost_mb and image_cost_mb parameters. Default values are 200MB for
        the text prompt and 150MB per image.

    Args:
    - batch_items: List of input items, each containing "conversations" and optional
        "image"/"video" keys.
    - model_path: Path to the ProcVLM checkpoint directory.
    - max_new_tokens, temperature: Generation parameters.
    - device_map: Passed to model loading for device placement. Can be "auto" or
        specific device string.
    - torch_dtype: Passed to model loading for dtype. Can be "auto" or specific dtype
        string.
    - dp: Degree of data parallelism for inference. If >1 and multiple GPUs are
        available, the batch will be split and processed in parallel across GPUs. If
        1, runs on a single device.
    - enable_value_head: Whether to run the value head prediction. If True, progress
        values will be predicted and injected into the generated text.
    - base_cost_mb, image_cost_mb: Estimated VRAM cost in MB for the base model and
        per image, used for dynamic batching. Adjust based on your workload. Defaults
        are empirically derived from profiling Qwen3VL with 2000 tokens and 480x480
        images. """
    if not batch_items:
        return []
    show_progress = True

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    worker_count = min(max(int(dp), 1), available_gpus if available_gpus > 0 else 1, len(batch_items))

    # Single worker optimization
    if worker_count <= 1:
        local_device = device_map if device_map != "auto" else ("cuda:0" if available_gpus > 0 else "cpu")
        model, processor = load_procvlm(model_path, local_device, torch_dtype)
        return model.batch_infer(
            batch_items=batch_items,
            processor=processor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )

    # Multi-worker DP execution
    worker_devices = [f"cuda:{i}" for i in range(worker_count)]

    # Preload model per device serially to avoid concurrent loading contention.
    for dev in worker_devices:
        load_procvlm(model_path, dev, torch_dtype)

    if show_progress:
        pbar = tqdm(total=len(batch_items), desc="Inference with Value Head", unit="item")

    chunk_size = (len(batch_items) + worker_count - 1) // worker_count
    results: List[str] = [""] * len(batch_items)

    def _worker_fn(worker_idx: int):
        start = worker_idx * chunk_size
        end = min(start + chunk_size, len(batch_items))
        if start >= end: return None
        
        local_batch = batch_items[start:end]
        local_device = worker_devices[worker_idx]
        
        # Threads on different GPUs load in parallel
        model, processor = load_procvlm(model_path, local_device, torch_dtype)
        preds = model.batch_infer(
            batch_items=local_batch,
            processor=processor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )
        return start, preds

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        # Map indices to ensure order
        futures = [executor.submit(_worker_fn, i) for i in range(worker_count)]
        for f in futures:
            res = f.result()
            if res:
                s, p = res
                results[s:s+len(p)] = p
                if show_progress:
                    pbar.update(len(p))
                    
    if show_progress:
        pbar.close()
    return results


# ===============================================
# Model Loading Helpers, Internal Usage Only
# ===============================================

_PROCVLM_HEAD_RELATIVE_PATH = Path("procvlm_extra") / "procvlm_head.pt"
_PROCVLM_HEAD_CANDIDATES = (
    _PROCVLM_HEAD_RELATIVE_PATH,
    Path("procvlm_head.pt"),
    Path("procvlm_head.bin"),
)

_PROCVLM_CACHE: Dict[Tuple[str, str, str], Tuple["ProcVLMWithValueHead", Any]] = {}
_PROCVLM_CACHE_LOCK = threading.Lock()
_PROCVLM_INFLIGHT_LOCKS: Dict[Tuple[str, str, str], threading.Lock] = {}
_PROCVLM_INFLIGHT_LOCKS_GUARD = threading.Lock()


def _parse_torch_dtype(dtype_spec: str | torch.dtype) -> torch.dtype | None:
    if isinstance(dtype_spec, torch.dtype):
        return dtype_spec
    if not isinstance(dtype_spec, str):
        return None

    value = dtype_spec.lower()
    if value == "auto":
        return None
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32", "float"}:
        return torch.float32
    return None


def _resolve_effective_dtype(
    torch_dtype: str | torch.dtype,
    attn_implementation: str,
) -> torch.dtype | None:
    dtype = _parse_torch_dtype(torch_dtype)
    if dtype is not None:
        return dtype

    # Flash Attention 2 expects an explicit half precision dtype.
    if attn_implementation == "flash_attention_2":
        return torch.bfloat16
    return None


def _resolve_target_device(device_map: str) -> str:
    if device_map == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device_map


def _remap_legacy_checkpoint_key(key: str) -> str:
    # Collapse nested legacy ProcVLM wrapper prefixes.
    while key.startswith("base_model."):
        key = key.replace("base_model.", "", 1)

    # Repair legacy duplicated model namespace.
    while key.startswith("model.model."):
        key = key.replace("model.", "", 1)

    # Normalize legacy model.lm_head.* to lm_head.*
    if key.startswith("model.lm_head."):
        key = key.replace("model.", "", 1)

    return key


def _is_procvlm_state_dict(state_dict: Dict[str, torch.Tensor]) -> bool:
    if not isinstance(state_dict, dict) or not state_dict:
        return False
    return any(
        key.startswith("base_model.")
        or key.startswith("pooler.")
        or key.startswith("value_head.")
        for key in state_dict.keys()
    )


def _has_lora_adapter(model_path: str) -> bool:
    path = Path(model_path)
    return path.is_dir() and (path / "adapter_config.json").exists()


def _load_procvlm_head_state(model_path: str) -> Dict[str, torch.Tensor] | None:
    path = Path(model_path)
    state_dict = None
    for candidate in _PROCVLM_HEAD_CANDIDATES:
        head_path = path / candidate
        if not head_path.exists():
            continue
        try:
            state_dict = torch.load(head_path, map_location="cpu")
            break
        except Exception:
            continue

    if state_dict is None:
        return None

    if not isinstance(state_dict, dict):
        return None

    adapted_state: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        remapped = _remap_legacy_checkpoint_key(key)
        if remapped.startswith("pooler.") or remapped.startswith("value_head."):
            adapted_state[remapped] = value

    if adapted_state:
        return adapted_state
    return None


def _split_base_and_head_state(
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Split a state dict into base-model weights and ProcVLM head weights.

    Keys are remapped from legacy namespaces to current native namespaces first.
    """
    base_state: Dict[str, torch.Tensor] = {}
    head_state: Dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        remapped = _remap_legacy_checkpoint_key(key)
        if remapped.startswith("pooler.") or remapped.startswith("value_head."):
            head_state[remapped] = value
        else:
            base_state[remapped] = value

    return base_state, head_state


def _load_procvlm_state_dict(model_path: str) -> Dict[str, torch.Tensor] | None:
    path = Path(model_path)

    procvlm_bin = path / "procvlm_model.bin"
    if procvlm_bin.exists():
        state_dict = torch.load(procvlm_bin, map_location="cpu")
        return state_dict if _is_procvlm_state_dict(state_dict) else None

    single_bin = path / "pytorch_model.bin"
    if single_bin.exists():
        state_dict = torch.load(single_bin, map_location="cpu")
        return state_dict if _is_procvlm_state_dict(state_dict) else None

    index_file = path / "pytorch_model.bin.index.json"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        weight_map = index_data.get("weight_map", {})
        if not isinstance(weight_map, dict):
            return None

        robot_keys = [
            key for key in weight_map.keys()
            if key.startswith("base_model.")
            or key.startswith("model.")
            or key.startswith("pooler.")
            or key.startswith("value_head.")
        ]
        if not robot_keys:
            return None

        shards_to_load = sorted({weight_map[key] for key in robot_keys})
        merged_state: Dict[str, torch.Tensor] = {}
        for shard_name in shards_to_load:
            shard_path = path / shard_name
            shard_state = torch.load(shard_path, map_location="cpu")
            merged_state.update(shard_state)

        return merged_state if _is_procvlm_state_dict(merged_state) else None

    return None


def _build_qwen_messages(batch_items: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    messages_list = []
    for item in batch_items:
        content = []

        images = item.get("image", [])
        if isinstance(images, str):
            images = [images]
        for img in images:
            content.append({"type": "image", "image": img})

        videos = item.get("video", [])
        if isinstance(videos, str):
            videos = [videos]
        for vid in videos:
            content.append({"type": "video", "video": vid})

        query = item["conversations"][0]["value"]
        query = query.replace("<image>", "").replace("<video>", "").strip()

        content.append({"type": "text", "text": query})
        messages_list.append([
            {"role": "user", "content": content}
        ])
    return messages_list


def _prepare_chunk_inputs(chunk_items, processor, need_per_sample_visual=False):
    """Process raw batch items into CPU tensors ready for GPU transfer.

    Returns ``(inputs, per_sample_visual_kwargs)``.  ``per_sample_visual_kwargs``
    is an empty list when *need_per_sample_visual* is False.
    """
    messages_list = _build_qwen_messages(chunk_items)
    texts = [
        processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_list
    ]
    image_inputs, video_inputs = process_vision_info(messages_list)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    per_sample_visual_kwargs: List[Dict[str, torch.Tensor]] = []
    if need_per_sample_visual:
        for i in range(len(texts)):
            sample_messages = [messages_list[i]]
            sample_images, sample_videos = process_vision_info(sample_messages)
            sample_inputs = processor(
                text=[texts[i]],
                images=sample_images,
                videos=sample_videos,
                padding=True,
                return_tensors="pt",
            )
            per_sample_visual_kwargs.append({
                k: v for k, v in sample_inputs.items()
                if k in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw")
            })

    return inputs, per_sample_visual_kwargs
def _safe_device_context(device_str: str):
    """Context manager for enforcing parameter allocation to the right device natively."""
    if device_str is None or device_str == "auto":
        yield
        return

    # Leveraging PyTorch 2.0+ context manager (fully supported by Qwen requirements)
    if hasattr(torch.device, "__enter__"):
        with torch.device(device_str):
            yield
    else:
        yield


def _global_process_worker(args):
    s, e, chunk_items, model_path, max_new_tokens, temperature, sampling_kwargs = args
    
    from core.models.qwenvl import process_batch_chat_vllm
    
    processed = process_batch_chat_vllm(
        batch_items=chunk_items,
        model_path=model_path,
        max_tokens=max_new_tokens,
        temperature=temperature,
        sampling_kwargs=sampling_kwargs,
    )
    return s, e, processed