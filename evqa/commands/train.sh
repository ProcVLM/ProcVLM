#!/bin/bash
# Complete QwenVL Training Launch Script with Full Parameter Documentation

# ======================
# Distributed Configuration
# ======================
MASTER_ADDR="127.0.0.1"                     # [Required] Master node IP for multi-GPU training
MASTER_PORT=$(shuf -i 20000-29999 -n 1)     # Random port to avoid conflicts
# NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  # Automatically detects available GPUs
NPROC_PER_NODE=4

# ======================
# Path Configuration
# ======================
MODEL_PATH=$VLM_PATH_QWEN_TRAIN  # [ModelArguments] Pretrained model path
OUTPUT_DIR="./output/checkpoints"                   # Directory for saving checkpoints
CACHE_DIR="./tmp"                          # [TrainingArguments] Cache directory for models

# ======================
# Model Configuration
# ======================
DATASETS="procedural_40b_train%100,procedural_40b_test%100"                  # [DataArguments] Dataset with sampling rate
VAL_DATASETS="procedural_40b_test%20,procedural_40b_valid%50"                       # [DataArguments] Validation dataset with sampling rate

# ======================
# Training Hyperparameters
# ======================
torchrun --nproc_per_node=$NPROC_PER_NODE \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         evqa/train/train_qwen.py \
         # Core Arguments
         --model_name_or_path $MODEL_PATH \  # [ModelArguments] Model identifier
         --tune_mm_llm True \                # [TrainingArguments] Train LLM or not
         --tune_mm_vision True \             # [TrainingArguments] Train VIT or not
         --tune_mm_mlp True \                # [TrainingArguments] Train MLP or not
         --dataset_use $DATASETS \           # [DataArguments] Dataset specification
         --val_dataset_use $VAL_DATASETS \   # [DataArguments] Validation dataset specification
         --output_dir $OUTPUT_DIR \          # Output directory for checkpoints
         --cache_dir $CACHE_DIR \            # [TrainingArguments] Model cache location

         # Precision & Memory
         --bf16 \                            # Use bfloat16 precision (Ampere+ GPUs)
         --per_device_train_batch_size 4 \   # Batch size per GPU
         --gradient_accumulation_steps 16 \  # Effective batch size multiplier

         # Learning Rate Configuration
         --learning_rate 1e-5 \              # Base learning rate
         --mm_projector_lr 2e-5 \            # [TrainingArguments] Projector-specific LR
         --vision_tower_lr 5e-6 \            # [TrainingArguments] Vision encoder LR
         --optim adamw_torch \               # [TrainingArguments] Optimizer selection

         # Sequence Configuration
         --model_max_length 131072 \         # [TrainingArguments] Max sequence length (128K Context)
         --data_flatten False \              # [DataArguments] Concatenate batch sequences
         --data_packing False \              # [DataArguments] Using packing data

         # Image Processing
         --max_pixels $((768*768)) \               # [DataArguments] Max image pixels (H*W) for image
         --min_pixels $((64*64)) \                # [DataArguments] Min image pixels for image
         # Video Processing
         --video_fps 2 \                          # [DataArguments] video fps
         --video_max_frames 512 \                   # [DataArguments] Max frames per video
         --video_min_frames 2 \                   # [DataArguments] Min frames per video
         --video_max_pixels $((768*768)) \        # [DataArguments] Max pixels per video
         --video_min_pixels $((64*64)) \         # [DataArguments] Min pixels per video

         # Training Schedule
         --num_train_epochs 3 \              # Total training epochs
         --warmup_ratio 0.03 \               # LR warmup proportion
         --lr_scheduler_type "cosine" \      # Learning rate schedule
         --weight_decay 0.01 \               # L2 regularization strength

         # Logging & Checkpoints
         --logging_steps 10 \               # Log metrics interval
         --save_steps 500 \                 # Checkpoint save interval
         --save_total_limit 3 \             # Max checkpoints to keep
         --report_to tensorboard \          # Reporting tool for logs
         --logging_dir ./logs \             # Tensorboard log directory

         # Evaluation
         --eval_strategy "steps" \           # set to "steps" or "epoch"
         --eval_steps 500 \                  # Evaluate every 500 steps (recommended to align with save_steps)
         --metric_for_best_model "loss" \    # (Optional) Metric to use for saving the best model
         --load_best_model_at_end False \    # (Optional) Whether to load the best model at the end of training; usually set to False for large models to save time/space

         # Lora Config
         --lora_enable False \                 # [TrainingArguments] Enable LoRA

         # Advanced Options
         --deepspeed configs/finetune/ds_config_zero3.json           # DeepSpeed configuration