## Step 0: config dataset paths and gloabl DP size (world size)
# All datasets should be converted to LeRobot v2.1 format. We use a lightweight built-in reader based on LeRobot v2.1 protocol.
export DATASETS="\
/path/to/dataset1 \
/path/to/dataset2 \
/path/to/dataset3"


## Step 1: annotate subtasks
python tools/task_borders.py --dataset_paths $DATASETS --yes
python ecot/run_plan_task.py --dataset_paths $DATASETS
python ecot/run_sub_task.py --dataset_paths $DATASETS
# Postprocess and archive
python tools/archive_jsonl.py --category subtask --name subtask_v2 --dataset_paths $DATASETS
python ecot/postprocess/flatten_sub_task.py --src /pretrain_data/archived_jsonl/subtask_v2/ --dst /pretrain_data/annotation/v2/sub_task/ --dataset_paths $DATASETS


## Step 2: annotate grounding boxes
CUDA_VISIBLE_DEVICES=0,1 python grounding/run_internvl_pipeline.py --world_size 4 --rank 0 --dataset_paths $DATASETS &
CUDA_VISIBLE_DEVICES=2,3 python grounding/run_internvl_pipeline.py --world_size 4 --rank 1 --dataset_paths $DATASETS &
CUDA_VISIBLE_DEVICES=4,5 python grounding/run_internvl_pipeline.py --world_size 4 --rank 2 --dataset_paths $DATASETS &
CUDA_VISIBLE_DEVICES=6,7 python grounding/run_internvl_pipeline.py --world_size 4 --rank 3 --dataset_paths $DATASETS &
wait
# Postprocess and archive
# grounding/postprocess/clean_bbox_stage3.py uses CUDA to accelerate matrix operation
python grounding/postprocess/flatten_bbox.py --dataset_paths $DATASETS
CUDA_VISIBLE_DEVICES=0 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 0 --dataset_paths $DATASETS & 
CUDA_VISIBLE_DEVICES=0 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 1 --dataset_paths $DATASETS &
CUDA_VISIBLE_DEVICES=1 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 2 --dataset_paths $DATASETS &
CUDA_VISIBLE_DEVICES=1 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 3 --dataset_paths $DATASETS &
CUDA_VISIBLE_DEVICES=2 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 4 --dataset_paths $DATASETS &
CUDA_VISIBLE_DEVICES=2 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 5 --dataset_paths $DATASETS &
CUDA_VISIBLE_DEVICES=3 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 6 --dataset_paths $DATASETS &
CUDA_VISIBLE_DEVICES=3 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 7 --dataset_paths $DATASETS &
CUDA_VISIBLE_DEVICES=4 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 8 --dataset_paths $DATASETS &
CUDA_VISIBLE_DEVICES=4 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 9 --dataset_paths $DATASETS &
CUDA_VISIBLE_DEVICES=5 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 10 --dataset_paths $DATASETS &
CUDA_VISIBLE_DEVICES=5 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 11 --dataset_paths $DATASETS &
CUDA_VISIBLE_DEVICES=6 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 12 --dataset_paths $DATASETS &
CUDA_VISIBLE_DEVICES=6 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 13 --dataset_paths $DATASETS &
CUDA_VISIBLE_DEVICES=7 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 14 --dataset_paths $DATASETS &
CUDA_VISIBLE_DEVICES=7 python grounding/postprocess/clean_bbox_stage3.py --world_size 16 --rank 15 --dataset_paths $DATASETS &
wait
python tools/archive_jsonl.py --name v2 --category bbox --copy_only --dataset_paths $DATASETS


## Step 3: annotate reasoning
CUDA_VISIBLE_DEVICES=0,1 python ecot/run_cot_v2.py --world_size 4 --rank 0 --dataset_paths $DATASETS &
CUDA_VISIBLE_DEVICES=2,3 python ecot/run_cot_v2.py --world_size 4 --rank 1 --dataset_paths $DATASETS &
CUDA_VISIBLE_DEVICES=4,5 python ecot/run_cot_v2.py --world_size 4 --rank 2 --dataset_paths $DATASETS &
CUDA_VISIBLE_DEVICES=6,7 python ecot/run_cot_v2.py --world_size 4 --rank 3 --dataset_paths $DATASETS &
wait
# Postprocess and archive
python ecot/postprocess/flatten_cot.py --dataset_paths $DATASETS
python tools/archive_jsonl.py --name v2 --category cot --copy_only --dataset_paths $DATASETS