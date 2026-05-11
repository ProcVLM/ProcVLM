python grounding/postprocess/flatten_bbox.py
CUDA_VISIBLE_DEVICES=0 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 0 & 
CUDA_VISIBLE_DEVICES=0 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 1 &
CUDA_VISIBLE_DEVICES=0 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 2 &
CUDA_VISIBLE_DEVICES=0 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 3 &
CUDA_VISIBLE_DEVICES=1 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 4 &
CUDA_VISIBLE_DEVICES=1 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 5 &
CUDA_VISIBLE_DEVICES=1 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 6 &
CUDA_VISIBLE_DEVICES=1 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 7 &
CUDA_VISIBLE_DEVICES=2 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 8 &
CUDA_VISIBLE_DEVICES=2 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 9 &
CUDA_VISIBLE_DEVICES=2 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 10 &
CUDA_VISIBLE_DEVICES=2 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 11 &
CUDA_VISIBLE_DEVICES=3 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 12 &
CUDA_VISIBLE_DEVICES=3 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 13 &
CUDA_VISIBLE_DEVICES=3 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 14 &
CUDA_VISIBLE_DEVICES=3 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 15 &
CUDA_VISIBLE_DEVICES=4 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 16 &
CUDA_VISIBLE_DEVICES=4 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 17 &
CUDA_VISIBLE_DEVICES=4 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 18 &
CUDA_VISIBLE_DEVICES=4 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 19 &
CUDA_VISIBLE_DEVICES=5 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 20 &
CUDA_VISIBLE_DEVICES=5 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 21 &
CUDA_VISIBLE_DEVICES=5 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 22 &
CUDA_VISIBLE_DEVICES=5 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 23 &
CUDA_VISIBLE_DEVICES=6 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 24 &
CUDA_VISIBLE_DEVICES=6 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 25 &
CUDA_VISIBLE_DEVICES=6 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 26 &
CUDA_VISIBLE_DEVICES=6 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 27 &
CUDA_VISIBLE_DEVICES=7 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 28 &
CUDA_VISIBLE_DEVICES=7 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 29 &
CUDA_VISIBLE_DEVICES=7 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 30 &
CUDA_VISIBLE_DEVICES=7 python grounding/postprocess/clean_bbox_stage1.py --world_size 32 --rank 31 &
wait
CUDA_VISIBLE_DEVICES=0,1 python grounding/postprocess/clean_bbox_stage2.py --world_size 4 --rank 0 &
CUDA_VISIBLE_DEVICES=2,3 python grounding/postprocess/clean_bbox_stage2.py --world_size 4 --rank 1 &
CUDA_VISIBLE_DEVICES=4,5 python grounding/postprocess/clean_bbox_stage2.py --world_size 4 --rank 2 &
CUDA_VISIBLE_DEVICES=6,7 python grounding/postprocess/clean_bbox_stage2.py --world_size 4 --rank 3 &
wait
CUDA_VISIBLE_DEVICES=0 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 0 & 
CUDA_VISIBLE_DEVICES=0 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 1 &
CUDA_VISIBLE_DEVICES=1 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 2 &
CUDA_VISIBLE_DEVICES=1 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 3 &
CUDA_VISIBLE_DEVICES=2 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 4 &
CUDA_VISIBLE_DEVICES=2 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 5 &
CUDA_VISIBLE_DEVICES=3 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 6 &
CUDA_VISIBLE_DEVICES=3 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 7 &
CUDA_VISIBLE_DEVICES=4 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 8 &
CUDA_VISIBLE_DEVICES=4 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 9 &
CUDA_VISIBLE_DEVICES=5 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 10 &
CUDA_VISIBLE_DEVICES=5 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 11 &
CUDA_VISIBLE_DEVICES=6 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 12 &
CUDA_VISIBLE_DEVICES=6 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 13 &
CUDA_VISIBLE_DEVICES=7 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 14 &
CUDA_VISIBLE_DEVICES=7 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 15 &
wait
python archive_jsonl.py --name v2 --category bbox --copy_only
python grounding/postprocess/viz_video_bbox.py
