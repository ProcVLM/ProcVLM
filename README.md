## ProcVLM: Learning Procedure-Grounded Progress Rewards for Robotic Manipulation


### Project Structure

```
.
├── assets           # Documentation, images, and other resources
├── configs
│   └── finetune         # Deepspeed configuration files for fine-tuning
├── core             # Local inference engine and related utilities
│   ├── backends         # Local inference pipelines for different engines (e.g. vLLM, LMDeploy, SGLang, etc.)
│   ├── data             # Optimized data reader for LeRobot datasets
│   ├── models           # Higher level model.batch_infer() interface, user-friendly model loading and inference APIs
│   └── utils            # General utilities
├── ecot             # Pipelines for sub-task / reasoning (CoT) annotations, used for training data generation
├── envs             # pip environment files
├── evqa             # SFT / LoRA training / evaluation scripts for ProcVLM
│   ├── commands
│   ├── eval             # Evaluation and metrics scripts
│   ├── one-shot         # One-shot LoRA training scripts, including a GUI for sub-task annotation
│   └── train            # SFT scripts adapted from the original Qwen-VL codebase
├── grounding        # Pipelines for grounding (bounding box) annotations, used for training data generation
├── tests
└── tools
```

### Quick Start

In most cases, you can simply use uv to set up the environment:

```
# 1. Install uv: https://docs.astral.sh/uv/getting-started/installation/
wget -qO- https://astral.sh/uv/install.sh | sh

# 2. Set up the environment with Python 3.10
uv sync --python 3.10
source .venv/bin/activate

# 3. Install flash-attn with no-build-isolation flag
uv pip install flash-attn --no-build-isolation
```

This project uses vLLM v0.18 with Transformers v4.57 by default. If you encounter issues when using vLLM for inference, please refer to the [vLLM documentation](https://docs.vllm.ai/en/latest/getting_started/quickstart/) and [vLLM troubleshooting](https://docs.vllm.ai/en/stable/usage/troubleshooting/) guides to check the compatibility of your hardware and software environment. Reinstalling PyTorch with the appropriate CUDA version works in most cases.

#### Setup LMDeploy for Local Inference

For the grounding or reasoning annotation pipeline, LMDeploy must be set up for local inference. To set up LMDeploy, please refer to the [LMDeploy documentation](https://lmdeploy.readthedocs.io/en/latest/get_started/get_started.html) to check the compatibility of your hardware and software environment. We recommend using Conda to manage the LMDeploy environment separately.

```
# 1. Create and activate a new Conda environment
conda create -n lmdeploy python=3.10 -y
conda activate lmdeploy

# 2. Install LMDeploy with pip
pip install lmdeploy

# 3. Install other dependencies
pip install -r envs/others_pip.txt
```

### Progress Reward Inference

You can run progress reward inference on a given video and save frame-wise predictions to a JSONL file with:

```
python evqa/inference.py \
    --model_path /path/to/model \
    --video_path tmp/fold_cloth/R1_Lite_fold_clothes.mp4 \
    --output_path tmp/fold_cloth/R1_Lite_fold_clothes_progress.jsonl \
    --task "fold the red T-shirt" \
    --window_size 8
# window_size: the number of recent frames (including the current frame) to use for progress estimation
```

Each JSONL row contains one sampled `frame_index` and its corresponding `progress` prediction. You can also use the built-in `infer_progress_from_video()` API from `evqa/inference.py` to get the same prediction sequence directly in Python.

You can run progress visualization on a given video with:

```
python evqa/eval/visualize_progress_video \
    --model_path /path/to/model \
    --video_path tmp/fold_cloth/R1_Lite_fold_clothes.mp4 \
    --output_path tmp/fold_cloth/R1_Lite_fold_clothes_progress_vis.mp4 \
    --task "fold the red T-shirt" \
    --window_size 8
# window_size: the number of recent frames (including the current frame) to use for progress estimation
```

#### Alternative ProcVLM Inference Methods

We recommend running ProcVLM inference with vLLM as a standard Qwen-VL model. Since the input consists mostly of multi-image queries, we provide a useful API at `evqa.model.batch_chat_with_vllm()` to handle image processing and asynchronous engine generation. The CPU processing and GPU inference stages are well pipelined through multiprocessing, so you can simply pass in large batches of queries. The API is defined as:

```python
batch_chat_with_vllm(
    batch_items: List[Dict[str, Any]],
        # Each item should contain a "conversations" key and optionally 
        #   "image" / "video" keys.
    model_path: str,
        # Path to the ProcVLM checkpoint directory.
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    tp: int = 1,
    sampling_kwargs: Optional[Dict] = None,
        # "top_p", "top_k", etc. passed to vLLM's generate() function.
    engine_kwargs: Optional[Dict] = None
        # "dtype", "gpu_id", etc. for vLLM engine creation.
        # Pass "process_workers" to control the number of parallel CPU 
        #   workers for preprocessing (default 16).
        # Pass "pipeline_chunk_size" to control how many items each 
        #   worker processes per batch (default 8).
) -> List[str]:
    # Returns a list of generated answers (plain text) in the same order
    #   as batch_items.
    ...
```

To run evaluation on ProcVLM, first modify the dataset config in `./evqa/data/__init__.py`:

```python
"test": {
    "annotation_path": "procvqa-50m-20260324/test/full.jsonl",  # Path to annotation file (jsonl format)
    "data_path": "procvqa-50m-20260324/"                        # Path to media resources (images/videos)
},
```

Then run the evaluation script:

```bash
# set batch_size=-1 to use the API's internal dynamic batching
python evqa/eval/run_eval.py \
    --model_type procvlm \
    --model_path /path/to/your/model \
    --dataset_use "test,<other_datasets>" \
    --batch_size -1 \
    --log_dir /path/to/save/logs
```

You can also chat with the model via the Chat CLI:

```
python evqa/chat_cli \
    --model-path /path/to/your/checkpoint \
    --device-map auto \
    --torch-dtype bf16 \
    --max-new-tokens 1024 \
    --temperature 0.1
```

### LoRA Fine-tuning

You can adapt ProcVLM to a new environment with only one successful task demonstration. See `evqa/docs/oneshot_adaptation.md` for how to build a one-shot fine-tuning dataset from one or a few demonstration videos and launch LoRA training.

### Running the Pipelines

Before running the pipelines, please make sure to set up the local inference engines. Then, following `run_ecot_all.sh` to run the full annotation pipeline.
