CUDA_VISIBLE_DEVICES=0 python evqa/eval/visualize_progress_video.py \
    --model_path /path/to/model \
    --video_path tmp/fold_cloth/R1_Lite_fold_clothes.mp4 \
    --output_path tmp/fold_cloth/R1_Lite_fold_clothes_progress_vis.mp4 \
    --task "fold the red T-shirt" \
    --window_size 4