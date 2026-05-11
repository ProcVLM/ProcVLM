CUDA_VISIBLE_DEVICES=0,1 python grounding/run_internvl_pipeline.py --world_size 4 --rank 0 --test_run &
CUDA_VISIBLE_DEVICES=2,3 python grounding/run_internvl_pipeline.py --world_size 4 --rank 1 --test_run &
CUDA_VISIBLE_DEVICES=4,5 python grounding/run_internvl_pipeline.py --world_size 4 --rank 2 --test_run &
CUDA_VISIBLE_DEVICES=6,7 python grounding/run_internvl_pipeline.py --world_size 4 --rank 3 --test_run &
wait
python grounding/postprocess/flatten_bbox_for_test.py
CUDA_VISIBLE_DEVICES=0 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 0 --test_run & 
CUDA_VISIBLE_DEVICES=0 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 1 --test_run &
CUDA_VISIBLE_DEVICES=1 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 2 --test_run &
CUDA_VISIBLE_DEVICES=1 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 3 --test_run &
CUDA_VISIBLE_DEVICES=2 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 4 --test_run &
CUDA_VISIBLE_DEVICES=2 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 5 --test_run &
CUDA_VISIBLE_DEVICES=3 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 6 --test_run &
CUDA_VISIBLE_DEVICES=3 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 7 --test_run &
CUDA_VISIBLE_DEVICES=4 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 8 --test_run &
CUDA_VISIBLE_DEVICES=4 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 9 --test_run &
CUDA_VISIBLE_DEVICES=5 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 10 --test_run &
CUDA_VISIBLE_DEVICES=5 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 11 --test_run &
CUDA_VISIBLE_DEVICES=6 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 12 --test_run &
CUDA_VISIBLE_DEVICES=6 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 13 --test_run &
CUDA_VISIBLE_DEVICES=7 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 14 --test_run &
CUDA_VISIBLE_DEVICES=7 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 15 --test_run &
wait
# remember to change test_name accordingly
python grounding/postprocess/viz_video_bbox_for_test.py --test_name intern35_v2b21
