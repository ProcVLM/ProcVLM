export CUDA_VISIBLE_DEVICES=0                                  # Set visible GPUs
MODEL_PATH=$VLM_PATH_QWEN_TRAIN                                # [ModelArguments] Pretrained model path
DATASETS="train_single_task_oneshot%100"                       # [DataArguments] Dataset with sampling rate

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="path/to/checkpoints/oneshot_lora_${TIMESTAMP}"     # Directory for saving checkpoints
LOG_DIR="./logs/qwen3vl/oneshot_lora_${TIMESTAMP}"             # Directory for TensorBoard logs
CACHE_DIR="./tmp"

python evqa/train/train_qwen.py \
    --model_name_or_path $MODEL_PATH \
    --tune_mm_llm False \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --dataset_use $DATASETS \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --bf16 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --optim adamw_torch \
    --model_max_length 8192 \
    --data_flatten False \
    --data_packing False \
    --dataloader_num_workers 2 \
    --max_pixels $((512*512)) \
    --min_pixels $((32*32)) \
    --video_fps 2 \
    --video_max_frames 256 \
    --video_min_frames 1 \
    --video_max_pixels $((512*512)) \
    --video_min_pixels $((32*32)) \
    --num_train_epochs 3 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "linear" \
    --weight_decay 0.01 \
    --logging_steps 1 \
    --save_steps 20 \
    --save_total_limit 1 \
    --report_to tensorboard \
    --logging_dir $LOG_DIR \
    --eval_strategy "no" \
    --metric_for_best_model "loss" \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --gradient_checkpointing True \
    --value_loss_weight 0.1 \
    --value_dropout 0.05 \
    --value_noise_std 0.02 2>&1 | tee logs/evqa_train_oneshot_lora.log

    # If you think it's necessary, use the argument below to enable deepspeed
    #   for faster training and better memory efficiency.
    # --deepspeed configs/finetune/ds_config_zero2.json \