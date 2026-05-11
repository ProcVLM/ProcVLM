python evqa/tools/split_datasets.py \
    --dataset_paths \
        /path/to/dataset1 \
        ... \
    --valid_datasets \
        /path/to/valid_dataset1 \
        ... \
    --selected_test_file /path/to/selection.jsonl \
    --input_root /path/to/input_root \
    --output_root /path/to/output_root \
    --train_ratio 0.95 \
    --test_ratio 0.00 \
    --valid_ratio 0.05
