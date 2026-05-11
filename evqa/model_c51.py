import torch
import torch.nn as nn
import re
import json
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import contextlib

from qwen_vl_utils import process_vision_info
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
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


class ProcVLMWithValueHead(nn.Module):
    """ Procedural VLM with Value Head for Embodied Reasoning and Rewarding

    Adapted from Qwen3VL base, this model adds an attention pooling layer over 
    the full sequence of hidden states and a dedicated value head MLP to predict
    a scalar.
    
    The training objective combines ordinary CE loss for language modeling with 
    an auxiliary regression loss on the predicted value head output, supervised 
    by progress labels if present in the training data. Formally:

       L = L_CE (gt_text, pred_text) + λ * L_C51 (pred_progress, gt_progress)

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
            base_model, 
            tokenizer, 
            value_loss_weight=0.2, 
            value_dropout=0, 
            value_noise_std=0.03, 
            value_output_size=10
        ):
        """ Args:
        - base_model: 
            The underlying QwenVL model (e.g., Qwen3VLForConditionalGeneration).
        - tokenizer: 
            The corresponding tokenizer for decoding progress from labels.
        - value_loss_weight: 
            λ, the weight for the auxiliary value loss in the total loss.
        - value_dropout: 
            Dropout probability for the value head MLP to prevent overfitting and 
            encourage robustness. Two dropout layers are applied with same rate.
        - value_noise_std: 
            Standard deviation of Gaussian noise injected into pooled representa-
            tion during training for exposure bias mitigation.
        - value_loss_type: 
            Type of regression loss to use ("l1" or "smooth_l1").
        - value_output_size: 
            Size of soft-binning output for value head. Default 10 corresponds to
            10 bins of 10% progress each.
        """
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        config = base_model.config
        hidden_size = getattr(config, 'hidden_size', None) or config.text_config.hidden_size

        # attention pooling
        self.pooler = AttentionPooling(hidden_size)

        # value head (wider + deeper MLP for better regression capacity)
        self.value_head = nn.Sequential( # 20M params for hidden_size=1536
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(value_dropout),
            nn.Linear(hidden_size // 2, value_output_size),
        )
        self.value_output_size = value_output_size

        # hyperparameters for value loss computation
        self.value_loss_weight = value_loss_weight
        
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
        self.lm_head = self.base_model.get_output_embeddings()
    

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
        model_cls = _resolve_qwen_vl_model_class(model_path)
        procvlm_state = _load_procvlm_state_dict(model_path)
        target_dtype = _resolve_effective_dtype(torch_dtype, attn_implementation)
        target_device = _resolve_target_device(device_map)

        if procvlm_state is not None:
            # Load directly from internal config with manually isolated state_dict
            # Avoid passing state_dict to from_pretrained directly to bypass ValueError
            base_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            if hasattr(base_config, "_attn_implementation"):
                base_config._attn_implementation = attn_implementation
            else:
                setattr(base_config, "_attn_implementation", attn_implementation)

            base_state = {
                key.replace("base_model.", "", 1): value
                for key, value in procvlm_state.items()
                if key.startswith("base_model.")
            }

            # Enforce parameters (e.g., Qwen3 RoPE inv_freq buffer caches) to be born securely on target_device
            # This completely patches 'Expected all tensors to be on same device cpu/cuda' natively.
            with _safe_device_context(target_device):
                base_model = model_cls(base_config)            
            # Tensors natively loaded inside the previously established device mapping, strict=False safely bypasses missing keys
            base_model.load_state_dict(base_state, strict=False)
            # Cast LayerNorm/Linear layers precisely
            if target_dtype is not None:
                base_model = base_model.to(dtype=target_dtype)

            # Assign and initialize the ValueHead layers inside device context as well
            with _safe_device_context(target_device):
                model = cls(base_model=base_model, tokenizer=tokenizer, **kwargs)
            if target_dtype is not None:
                model = model.to(dtype=target_dtype)

            # Finally, load the value head weights
            head_state = {
                key: value for key, value in procvlm_state.items() if not key.startswith("base_model.")
            }
            model.load_state_dict(head_state, strict=False)
            model.eval()
            return model

        # Standard Clean Hugging Face Loading
        load_kwargs = dict(trust_remote_code=True)
        if target_dtype is not None:
            load_kwargs["torch_dtype"] = target_dtype
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        if hasattr(model_cls, "from_pretrained") and "attn_implementation" in model_cls.from_pretrained.__code__.co_varnames:
            load_kwargs["attn_implementation"] = attn_implementation

        load_kwargs.update(kwargs)
        base_model = model_cls.from_pretrained(model_path, **load_kwargs)
        
        # Safely hook and align remaining value modules using mapped environments
        with _safe_device_context(target_device):
            model = cls(base_model=base_model, tokenizer=tokenizer, **kwargs)
        if target_dtype is not None:
            model = model.to(dtype=target_dtype)
        model.eval()
        return model


    def save_pretrained(
        self, 
        output_dir: str, 
        state_dict: Dict[str, torch.Tensor] | None = None
    ):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.base_model.config.save_pretrained(output_path)
        if state_dict is None:
            state_dict = self.state_dict()
        torch.save(state_dict, output_path / "pytorch_model.bin")
    

    # ------------------------------------------------
    # properties
    # ------------------------------------------------
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)


    # ------------------------------------------------
    # utilities
    # ------------------------------------------------
    def _ensure_value_head_device(self, device: torch.device):
        """ Keep value modules on the same device as hidden states during model-parallel inference. """
        if next(self.pooler.parameters()).device != device:
            self.pooler.to(device)
        if next(self.value_head.parameters()).device != device:
            self.value_head.to(device)


    def _chunk_batch_by_vram(
        self, 
        batch_items: List[Dict[str, Any]], 
        device: torch.device, 
        base_cost_mb: int,
        image_cost_mb: int,
    ) -> List[Tuple[List[int], List[Dict[str, Any]]]]:
        """ Dynamically chunk batch items based on available VRAM and a cost heuristic. """
        if device.type != "cuda":
            return [(list(range(len(batch_items))), batch_items)]

        free_vram, _ = torch.cuda.mem_get_info(device)
        # Leave a 15% safety margin for activations and other overhead
        safe_free_mb = (free_vram / (1024 ** 2)) * 0.85

        chunks = []
        current_chunk = []
        current_indices = []
        current_cost_mb = 0.0

        for idx, item in enumerate(batch_items):
            imgs = item.get("image", [])
            if isinstance(imgs, str):
                imgs = [imgs]
            
            item_cost_mb = base_cost_mb + len(imgs) * image_cost_mb

            if current_chunk and (current_cost_mb + item_cost_mb > safe_free_mb):
                chunks.append((current_indices, current_chunk))
                current_chunk = []
                current_indices = []
                current_cost_mb = 0.0

            current_chunk.append(item)
            current_indices.append(idx)
            current_cost_mb += item_cost_mb

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

    
    def _logits_to_values(self, value_logits: torch.Tensor) -> torch.Tensor:
        """ Convert discrete bins to continuous progress values in [0, 1] via weighted sum of bin centers. """
        num_bins = value_logits.shape[-1]
        if torch.allclose(value_logits.sum(dim=-1), torch.tensor(1.0, device=value_logits.device, dtype=value_logits.dtype), atol=1e-3):
            bin_probs = value_logits
        else:
            bin_probs = (
                torch.softmax(value_logits, dim=-1)
                if isinstance(value_logits, torch.Tensor)
                else np.softmax(value_logits, axis=-1)
            )
        bin_centers = (
            torch.linspace(0.0, 1.0, num_bins, device=value_logits.device, dtype=value_logits.dtype)
            if isinstance(value_logits, torch.Tensor)
            else np.linspace(0.0, 1.0, num_bins)
        )
        return (
            (bin_probs * bin_centers).sum(dim=-1)
            if isinstance(value_logits, torch.Tensor)
            else (bin_probs * bin_centers).sum(axis=-1)
        )
    
    
    def _values_to_soft_targets(self, values: torch.Tensor) -> torch.Tensor:
        """Convert continuous values in [0,1] to soft target probabilities."""
        num_bins = self.value_output_size
        device = values.device
        dtype = values.dtype

        # scale values to bin index space
        values = torch.clamp(values, 0.0, 1.0)
        scaled = values * (num_bins - 1)
        # left and right bin indices
        left_idx = torch.floor(scaled).long()
        right_idx = torch.clamp(left_idx + 1, max=num_bins - 1)
        
        # interpolation weights
        right_w = scaled - left_idx.to(scaled.dtype)
        left_w = 1.0 - right_w

        # construct target distribution
        probs = torch.zeros(values.shape[0], num_bins, device=device, dtype=dtype)
        probs.scatter_(1, left_idx.unsqueeze(-1), left_w.unsqueeze(-1))
        probs.scatter_add_(1, right_idx.unsqueeze(-1), right_w.unsqueeze(-1))

        return probs


    @torch.no_grad()
    def _predict_progress_values(self, generated_ids, full_hidden_states):
        """ Forward pass on seq hidden states → pooler → value_head → predicted progress [B]. (fp 0-100) """
        B, L_full, _ = full_hidden_states.shape
        attn_mask = torch.zeros((B, L_full), dtype=torch.long, device=generated_ids.device)
        for b in range(B):
            valid_mask = generated_ids[b] != self._pad_token_id
            if not valid_mask.any():
                continue
            first_valid = valid_mask.nonzero()[0].item()
            last_valid = valid_mask.nonzero()[-1].item()
            left_pad_count = first_valid
            right_pad_count = len(generated_ids[b]) - 1 - last_valid
            hs_start = left_pad_count
            hs_end = L_full - right_pad_count
            attn_mask[b, hs_start:hs_end] = 1

        # tail progress tag will not be masked during generation,
        # as it causes no harm at inference time

        self._ensure_value_head_device(full_hidden_states.device)

        pooler_dtype = next(self.pooler.parameters()).dtype
        hs_for_pooler = full_hidden_states.to(dtype=pooler_dtype)
        attn_mask_for_pooler = attn_mask.to(device=hs_for_pooler.device)

        pooled = self.pooler(hs_for_pooler, attn_mask_for_pooler)
        pooled = pooled.to(self.value_head[0].weight.dtype)

        value_logits = self.value_head(pooled) # [B, value_output_size]
        return self._logits_to_values(value_logits) * 100.0 # scale to [0, 100] for percentage


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
            
        value_logits = self.value_head(pooled) # [B, value_output_size]

        # No progress samples in this batch — return zero loss that still
        # keeps pooler/value_head in the computation graph so DeepSpeed
        # AllReduce doesn't deadlock across ranks.
        if not progress_mask.any():
            return (value_logits * 0).sum()

        # compute value loss 
        gt_probs = self._values_to_soft_targets(gt_progress / 100.0)
        value_loss_unreduced = nn.functional.cross_entropy(
            value_logits, gt_probs, reduction="none"
        )

        # Apply mask and reduce to scalar
        value_loss = (value_loss_unreduced * progress_mask.float()).sum() / (progress_mask.sum() + 1e-8)
        return value_loss.to(value_logits.dtype)


    def _compute_value_loss_packed(self, hidden_states, labels, attention_mask, progress_gt=None):
        """ Packed: labels [1, T], hidden_states [1, T, H], attention_mask [num_seqs+1] (cumsum).
        Split packed sequence into sub-sequences by boundaries, form a pseudo-batch,
        then reuse the same logic as _compute_value_loss_standard. """
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

        value_logits = self.value_head(pooled) # [num_seqs, value_output_size]

        # No progress samples — return zero loss that keeps params in graph
        if not progress_mask.any():
            return (value_logits * 0).sum()
        
        # compute value loss
        gt_probs = self._values_to_soft_targets(gt_progress / 100.0)
        value_loss_unreduced = nn.functional.cross_entropy(
            value_logits, gt_probs, reduction="none"
        )
        
        # Apply mask and reduce to scalar
        value_loss = (value_loss_unreduced * progress_mask.float()).sum() / (progress_mask.sum() + 1e-8)
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
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        progress_gt=None,
        **kwargs
    ):
        """ Forward pass with integrated value loss computation. """
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
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                position_ids=position_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
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
    def generate(self, return_value=False, *args, **kwargs):
        """ Generate with optional value-head forwarding. """
        # base model generation
        input_ids = kwargs.get('input_ids')
        if input_ids is None and args:
            input_ids = args[0]
        generated_ids = self.base_model.generate(*args, **kwargs)
        if not return_value: 
            return generated_ids
        # value head forwarding
        visual_kwargs = {
            k: v for k, v in kwargs.items()
            if k in ('pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw')
        }
        pred = self._predict_progress_values(generated_ids, **visual_kwargs)
        return generated_ids, pred


    @torch.no_grad()
    def infer(self, **inputs):
        """ Inference: generate text and replace progress with value-head prediction (bs=1). """
        generated_ids, pred = self.generate(**inputs, return_value=True, max_new_tokens=1024)
        text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        return text


    @torch.no_grad()
    def batch_infer(
        self,
        batch_items: List[Dict[str, Any]],
        processor,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        enable_value_head: bool = False,
        base_cost_mb: Optional[int] = 250,
        image_cost_mb: Optional[int] = 150,
        **generate_kwargs,
    ) -> List[str]:
        """ Args:
        - enble_value_head:
            Whether to run the value head and replace <progress>XX.XX%</progress> tags in the generated 
            text with the predicted values. This is default disabled to leverage the VLM's powerful
            generalization ability.
        - base_cost_mb, image_cost_mb:
            Estimated VRAM cost per image in MB for dynamic batching. Adjust based on the image size in
            your workload. If not specified, defaults to 250MB for base model and 150MB per image based 
            on empirical profiling with Qwen3VL (2000 tokens, 480x480 images).
        """
        if not batch_items:
            return []

        if processor.tokenizer.padding_side != "left":
            processor.tokenizer.padding_side = "left"

        device = next(self.base_model.parameters()).device
        model_dtype = next(self.base_model.parameters()).dtype

        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation_config["temperature"] = temperature
            generation_config["top_p"] = generate_kwargs.pop("top_p", 0.9)
        generation_config.update(generate_kwargs)

        # auto chunking by empirically estimated VRAM cost
        chunks = self._chunk_batch_by_vram(batch_items, device, base_cost_mb, image_cost_mb)
        
        # maxsize=2 prevents RAM OOM while perfectly hiding CPU processing latency
        data_queue = queue.Queue(maxsize=2)
        
        def producer():
            for chunk_indices, chunk_items in chunks:
                try:
                    messages_list = _build_qwen_messages(chunk_items)
                    texts = [
                        processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        for messages in messages_list
                    ]
                    image_inputs, video_inputs = process_vision_info(messages_list)
                    
                    # Creates CPU tensors
                    inputs = processor(
                        text=texts,
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    data_queue.put((chunk_indices, inputs, None))
                except Exception as e:
                    data_queue.put((None, None, e))
                    break
            data_queue.put((None, None, None)) # EOF Sentinel

        producer_thread = threading.Thread(target=producer)
        producer_thread.start()

        final_results = [""] * len(batch_items)

        while True:
            chunk_indices, inputs, exc = data_queue.get()
            if exc is not None:
                raise exc
            if chunk_indices is None:
                break

            # Consumer: Move to GPU exactly when needed
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    if v.is_floating_point():
                        inputs[k] = v.to(device=device, dtype=model_dtype)
                    else:
                        inputs[k] = v.to(device=device)

            if enable_value_head:
                captured_hs_list = []
                def generate_hook_fn(module, hook_args, output):
                    # hook_args[0] is the hidden_states before lm_head
                    # Prefill shape: [B, Prompt_Len, H]
                    # Decode shape per step: [B, 1, H]
                    captured_hs_list.append(hook_args[0].detach())
                handle = self.lm_head.register_forward_hook(generate_hook_fn)
                try:
                    generated_ids = self.generate(**inputs, **generation_config)
                finally:
                    handle.remove()
                # captured_hs_list: index 0 -> [B, Prompt_Len, H], index 1...N -> [B, 1, H] for each decoding step
                full_hidden_states = torch.cat(captured_hs_list, dim=1) # [B, total_len, H]
            else:
                generated_ids = self.generate(**inputs, **generation_config)
            
            prompt_lens = [len(ids) for ids in inputs.input_ids]
            generated_ids_trimmed = [g[l:] for g, l in zip(generated_ids, prompt_lens)]
            
            output_texts = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            if enable_value_head and any('<progress>' in text for text in output_texts): 
                pred = self._predict_progress_values(generated_ids, full_hidden_states)
                for i, text in enumerate(output_texts):
                    if '<progress>' in text:
                        output_texts[i] = self.progress_pattern.sub(
                            f'<progress>{pred[i].item():.2f}%</progress>', text,
                        )

            # Map back to global results
            for local_idx, global_idx in enumerate(chunk_indices):
                final_results[global_idx] = output_texts[local_idx]

            del inputs, generated_ids, generated_ids_trimmed
            if device.type == "cuda":
                # Helps prevent memory fragmentation across dynamic chunk sizes
                torch.cuda.empty_cache()

        producer_thread.join()
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

    if (path / "adapter_config.json").exists() and (path / "procvlm_head.bin").exists():
        return True

    index_file = path / "pytorch_model.bin.index.json"
    if index_file.exists():
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})
            if isinstance(weight_map, dict):
                return any(
                    key.startswith("base_model.") or key.startswith("pooler.") or key.startswith("value_head.")
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
    # First check without lock for high concurrency
    if cache_key in _ROBOT_VLM_CACHE:
        return _ROBOT_VLM_CACHE[cache_key]
    # Initialize outside the lock to allow parallel loading across different devices
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
    with _ROBOT_VLM_CACHE_LOCK:
        if cache_key not in _ROBOT_VLM_CACHE:
            _ROBOT_VLM_CACHE[cache_key] = (model, processor)
        return _ROBOT_VLM_CACHE[cache_key]


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
        the base_cost_mb and image_cost_mb parameters. Default values are 250MB for
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
    - base_cost_mb, image_cost_mb: Estimated VRAM cost in MB for the base model and 
        per image, used for dynamic batching. Adjust based on your workload. Defaults 
        are empirically derived from profiling Qwen3VL with 2000 tokens and 480x480 
        images. """
    if not batch_items:
        return []

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
    chunk_size = (len(batch_items) + worker_count - 1) // worker_count
    results: List[str] = [""] * len(batch_items)

    def _worker_fn(worker_idx: int):
        start = worker_idx * chunk_size
        end = min(start + chunk_size, len(batch_items))
        if start >= end: return None
        
        local_batch = batch_items[start:end]
        local_device = f"cuda:{worker_idx}"
        
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

    return results


# ===============================================
# Model Loading Helpers, Internal Usage Only
# ===============================================

_QWEN_VL_MODEL_CLS = {
    "Qwen2VLForConditionalGeneration": Qwen2VLForConditionalGeneration,
    "Qwen2_5_VLForConditionalGeneration": Qwen2_5_VLForConditionalGeneration,
    "Qwen3VLForConditionalGeneration": Qwen3VLForConditionalGeneration,
    "Qwen3VLMoeForConditionalGeneration": Qwen3VLMoeForConditionalGeneration,
}

_ROBOT_VLM_CACHE: Dict[Tuple[str, str, str], Tuple["ProcVLMWithValueHead", Any]] = {}
_ROBOT_VLM_CACHE_LOCK = threading.Lock()


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


def _is_procvlm_state_dict(state_dict: Dict[str, torch.Tensor]) -> bool:
    if not isinstance(state_dict, dict) or not state_dict:
        return False
    return any(
        key.startswith("base_model.") or key.startswith("pooler.") or key.startswith("value_head.")
        for key in state_dict.keys()
    )


def _has_lora_adapter(model_path: str) -> bool:
    path = Path(model_path)
    return path.is_dir() and (path / "adapter_config.json").exists()


def _load_procvlm_head_state(model_path: str) -> Dict[str, torch.Tensor] | None:
    path = Path(model_path)
    head_bin = path / "procvlm_head.bin"
    if not head_bin.exists():
        return None

    try:
        state_dict = torch.load(head_bin, map_location="cpu")
    except Exception:
        return None

    if not isinstance(state_dict, dict):
        return None

    if any(key.startswith("pooler.") or key.startswith("value_head.") for key in state_dict.keys()):
        return state_dict
    return None


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
            if key.startswith("base_model.") or key.startswith("pooler.") or key.startswith("value_head.")
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


def _resolve_qwen_vl_model_class(model_path: str):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    for arch in getattr(config, "architectures", []) or []:
        if arch in _QWEN_VL_MODEL_CLS:
            return _QWEN_VL_MODEL_CLS[arch]

    model_path_lower = model_path.lower()

    if "qwen3" in model_path_lower and "a" in model_path_lower:
        return Qwen3VLMoeForConditionalGeneration
    if "qwen3" in model_path_lower:
        return Qwen3VLForConditionalGeneration
    if "qwen2.5" in model_path_lower:
        return Qwen2_5_VLForConditionalGeneration
    if "qwen" in model_path_lower:
        return Qwen2VLForConditionalGeneration

    return AutoModelForImageTextToText


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


@contextlib.contextmanager
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