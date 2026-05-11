# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trainer import replace_qwen2_vl_attention_class, restore_qwen_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
from evqa.data.data_processor import make_supervised_data_module
from evqa.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoProcessor, Seq2SeqTrainer

# === Modified Qwen Model ===
from evqa.model import ProcVLMWithValueHead, is_procvlm_checkpoint

# === Custom Trainer for Qwen-VL Models ===
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers.trainer_utils import seed_worker
from typing import Optional, Callable, Any, Union

# Add custom Trainer to support generation during evaluation
# The retrieval tasks will be measured by generation-based metrics
class Qwen3VLTrainer(Seq2SeqTrainer):
    def __init__(self, *args, eval_data_collator, **kwargs):
        super().__init__(*args, **kwargs)
        # replace eval_data_collator, this is CRUCIAL when --predict_with_generate is enabled
        self.eval_data_collator = eval_data_collator if eval_data_collator else self.data_collator

    # --- Evaluation Dataloader without Data Packing ---
    def _get_dataloader_with_special_collator(
        self,
        data_collator: Callable,
        dataset: Dataset,
        description: str,
        batch_size: int,
        sampler_fn: Optional[Callable[[Dataset], torch.utils.data.Sampler]] = None,
        is_training: bool = False,
        dataloader_key: Optional[str] = None,
    ) -> DataLoader:
        data_collator = self._get_collator_with_removed_columns(data_collator, description=description)
        dataloader_params = {
            "batch_size": batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        if not isinstance(dataset, torch.utils.data.IterableDataset):
            if sampler_fn is not None:
                dataloader_params["sampler"] = sampler_fn(dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
            if is_training:
                dataloader_params["worker_init_fn"] = partial(
                    seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
                )
        dataloader = self.accelerator.prepare(DataLoader(dataset, **dataloader_params))
        # Store the prepared dataloader for subsequent evaluations if using persistent workers.
        if dataloader_key is not None and self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = dataloader
            else:
                self._eval_dataloaders = {dataloader_key: dataloader}
        return dataloader

    def get_eval_dataloader(self, eval_dataset=None):
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self._eval_dataloaders[dataloader_key]

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )

        return self._get_dataloader_with_special_collator(
            data_collator=self.eval_data_collator,
            dataset=eval_dataset,
            description="Evaluation",
            batch_size=self.args.eval_batch_size,
            sampler_fn=self._get_eval_sampler,
            dataloader_key=dataloader_key,
        )

    def get_test_dataloader(self, test_dataset):
        return self._get_dataloader_with_special_collator(
            data_collator=self.eval_data_collator,
            dataset=test_dataset,
            description="test",
            batch_size=self.args.eval_batch_size,
            sampler_fn=self._get_eval_sampler,
        )
    
    # --- Unified Generation Mode Switching ---
    def _switch_to_eval_mode(self):
        restore_qwen_attention_class()
        # record old states
        self._old_use_cache = self.model.config.use_cache
        self._old_padding_side = self.processing_class.padding_side
        # set eval
        self.model.config.use_cache = True
        self.processing_class.padding_side = "left"
        self.model.eval()

    def _switch_back_to_train_mode(self):
        replace_qwen2_vl_attention_class()
        # restore old states
        self.model.config.use_cache = self._old_use_cache
        self.processing_class.padding_side = self._old_padding_side
        self.model.train()

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        self._switch_to_eval_mode()
        try:
            results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        finally:
            self._switch_back_to_train_mode()
        return results

    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix="test", **gen_kwargs):
        self._switch_to_eval_mode()
        try:
            results = super().predict(test_dataset, ignore_keys, metric_key_prefix, **gen_kwargs)
        finally:
            self._switch_back_to_train_mode()
        return results
    
    # --- Log Value/LM Loss ---
    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        # retrieve auxiliary losses cached by ProcVLMWithValueHead.forward()
        raw_model = self.accelerator.unwrap_model(model)
        if hasattr(raw_model, '_last_value_loss'):
            self._value_loss_accum = getattr(self, '_value_loss_accum', 0.0) + raw_model._last_value_loss
            self._value_loss_count = getattr(self, '_value_loss_count', 0) + 1
        lm_loss = getattr(raw_model, '_last_lm_loss', None)
        if lm_loss is not None:
            self._lm_loss_accum = getattr(self, '_lm_loss_accum', 0.0) + lm_loss
            self._lm_loss_count = getattr(self, '_lm_loss_count', 0) + 1
        return loss

    def log(self, logs, *args, **kwargs):
        if hasattr(self, '_value_loss_count') and self._value_loss_count > 0:
            logs["value_loss"] = self._value_loss_accum / self._value_loss_count
            self._value_loss_accum = 0.0
            self._value_loss_count = 0
        if hasattr(self, '_lm_loss_count') and self._lm_loss_count > 0:
            logs["lm_loss"] = self._lm_loss_accum / self._lm_loss_count
            self._lm_loss_accum = 0.0
            self._lm_loss_count = 0
        return super().log(logs, *args, **kwargs)

    # --- Handle Max Generation Length ---
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # no generation fast path
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs)

        inputs = inputs.copy()

        # calculate loss before truncating inputs
        with torch.no_grad():
            loss = model(**inputs).loss
        
        # remove ground-truth labels from inputs to execute generation
        labels = inputs["labels"]
        is_label_mask = (labels != -100)
        if is_label_mask.any():
            B, T = labels.shape
            pad_token_id = self.processing_class.pad_token_id

            # Per-sample: keep only prompt tokens (before first label)
            prompt_ids_list = []
            for b in range(B):
                sample_mask = is_label_mask[b]
                if sample_mask.any():
                    first_label = sample_mask.nonzero(as_tuple=True)[0][0].item()
                else:
                    first_label = T
                prompt_ids_list.append(inputs["input_ids"][b, :first_label])

            # Left-pad to uniform length for batched generation
            max_prompt_len = max(p.shape[0] for p in prompt_ids_list)
            new_input_ids = torch.full(
                (B, max_prompt_len), pad_token_id,
                dtype=inputs["input_ids"].dtype, device=inputs["input_ids"].device,
            )
            new_attention_mask = torch.zeros(
                (B, max_prompt_len),
                dtype=inputs["attention_mask"].dtype, device=inputs["attention_mask"].device,
            )
            for b, prompt in enumerate(prompt_ids_list):
                L = prompt.shape[0]
                new_input_ids[b, max_prompt_len - L:] = prompt
                new_attention_mask[b, max_prompt_len - L:] = 1

            inputs["input_ids"] = new_input_ids
            inputs["attention_mask"] = new_attention_mask
            # Drop position_ids — model.generate recomputes from grid_thw + attention_mask
            inputs.pop("position_ids", None)
        raw_labels = labels
        inputs.pop("labels")
        inputs.pop("progress_gt", None)
        
        # set generation kwargs
        if self.args.max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = self.args.max_new_tokens
        gen_kwargs["num_beams"] = self.args.generation_num_beams if self.args.generation_num_beams else 1

        # generate sequences
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                **gen_kwargs,
                use_cache=True,
                pad_token_id=self.processing_class.pad_token_id,
                eos_token_id=self.processing_class.eos_token_id,
            )
        
        # Strip prompt prefix — keep only generated response tokens.
        # This keeps prediction_step behavior aligned with model.batch_infer.
        prompt_len = inputs["input_ids"].shape[1]
        generated_tokens = generated_tokens[:, prompt_len:]

        return loss, generated_tokens, raw_labels


# === Original code continues here ===
local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def safe_save_lora_and_value_head_for_inference(trainer: transformers.Trainer, output_dir: str, base_model_name_or_path: Optional[str] = None):
    """Optionally save LoRA adapter + ProcVLM value head for inference reload."""
    if not trainer.args.should_save:
        return

    model = trainer.model
    if hasattr(trainer, "accelerator"):
        model = trainer.accelerator.unwrap_model(model)

    if not hasattr(model, "peft_config"):
        return

    if base_model_name_or_path and hasattr(model, "peft_config"):
        for k in model.peft_config:
            model.peft_config[k].base_model_name_or_path = base_model_name_or_path

    # Save LoRA adapter in standard PEFT format.
    model.save_pretrained(output_dir)

    base_robot = None
    if hasattr(model, "get_base_model"):
        try:
            base_robot = model.get_base_model()
        except Exception:
            base_robot = None

    if base_robot is None and hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        base_robot = model.base_model.model

    if base_robot is None or not hasattr(base_robot, "pooler") or not hasattr(base_robot, "value_head"):
        logging.warning("LoRA adapter saved, but ProcVLM value head was not found for sidecar save.")
        return

    head_state = {
        **{f"pooler.{k}": v.cpu() for k, v in base_robot.pooler.state_dict().items()},
        **{f"value_head.{k}": v.cpu() for k, v in base_robot.value_head.state_dict().items()},
    }
    torch.save(head_state, Path(output_dir) / "procvlm_head.bin")
    logging.info("Saved LoRA adapter and ProcVLM value head sidecar for inference reload.")


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def is_main_process() -> bool:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    # --- Add new logging ---
    if 'tensorboard' in training_args.report_to and training_args.logging_dir:
        print(f"Report TensorBoard logs to {training_args.logging_dir}")
    
    # instantiate compute_metrics function
    try:
        from evqa.eval.compute_metrics import compute_accuracy_wrapper
        print(f"Using compute_metrics from {compute_accuracy_wrapper.__module__}.{compute_accuracy_wrapper.__name__}")
        def compute_metrics(eval_pred):
            return compute_accuracy_wrapper(eval_pred, tokenizer)
    except ImportError:
        compute_metrics = None
        print("No compute_metrics is used.")

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        use_fast=True
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    # --- Unified loading path via ProcVLMWithValueHead.from_pretrained ---
    procvlm_init = is_procvlm_checkpoint(model_args.model_name_or_path)
    if procvlm_init:
        print("Detected ProcVLM checkpoint. Loading with ProcVLM loader.")
    else:
        print("Loading base checkpoint with ProcVLM loader.")

    model = ProcVLMWithValueHead.from_pretrained(
        model_path=model_args.model_name_or_path,
        tokenizer=tokenizer,
        cache_dir=training_args.cache_dir,
        device_map="cpu",
        torch_dtype=(torch.bfloat16 if training_args.bf16 else "auto"),
        attn_implementation=attn_implementation,
        value_loss_weight=training_args.value_loss_weight,
        value_dropout=training_args.value_dropout,
        value_noise_std=training_args.value_noise_std,
    )
    data_args.model_type = "qwen3vl"

    print(f'the initlized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        if hasattr(model, "peft_config"):
            print("Model already contains a LoRA adapter. Reusing loaded adapter for training.")
            for p in model.parameters():
                p.requires_grad = False
            for n, p in model.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True
        else:
            for p in model.parameters():
                p.requires_grad = False

            lora_config = LoraConfig(
                r=training_args.lora_r or 64,
                lora_alpha=training_args.lora_alpha or 128,
                lora_dropout=training_args.lora_dropout or 0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 的 attention 线性层
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)
    else:
        set_model(model_args, model)

        if is_main_process():
            model.visual.print_trainable_parameters()
            model.model.print_trainable_parameters()

    # Value head is part of ProcVLMWithValueHead in unified loading path.
    print(f"Initialized ProcVLM model -> {model.__class__.__name__}")
    print("Enabling gradients for Value Head and Pooler...")
    for n, p in model.pooler.named_parameters():
        p.requires_grad = True
    for n, p in model.value_head.named_parameters():
        p.requires_grad = True
    if is_main_process():
        trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
        print(f"Total trainable parameters check: {len(trainable_params)}")
    # ============================
    
    data_module = make_supervised_data_module(processor, data_args=data_args)
    trainer = Qwen3VLTrainer(
        model=model, 
        processing_class=tokenizer, 
        args=training_args, 
        compute_metrics=compute_metrics,
        **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    if training_args.lora_enable:
        safe_save_lora_and_value_head_for_inference(trainer=trainer, output_dir=training_args.output_dir, base_model_name_or_path=model_args.model_name_or_path)
    
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
    # train(attn_implementation="sdpa")
