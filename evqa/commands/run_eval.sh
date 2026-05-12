python evqa/eval/run_eval.py \
    --model_type qwen \
    --model_path /path/to/your/model \
    --dataset_use "procedural_short_test,procedural_short_valid" \
    --batch_size 16 \
    --tp 8 \
    --log_dir /path/to/save/logs