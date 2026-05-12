---
language:
- en
pipeline_tag: image-text-to-text
library_name: transformers
tags:
- robotics
- vision-language-model
- progress-reward
- robot-manipulation
- qwen3-vl
- procvlm
license: apache-2.0
datasets:
- ce-amtic/ProcVQA-20M-annotations
base_model:
- Qwen/Qwen3-VL-2B-Instruct
---

# ProcVLM-2B

ProcVLM-2B is a procedure-grounded vision-language model for estimating progress rewards from robot manipulation observations. Given a task description and a recent window of video frames, the model reasons about the remaining atomic actions and predicts the current task completion percentage.

<p align="center">
  <a href="https://procvlm.github.io/">Homepage</a> |
  <a href="https://arxiv.org/abs/2605.08774">arXiv</a> |
  <a href="https://huggingface.co/ce-amtic/ProcVLM-2B">Model</a> |
  <a href="https://github.com/ProcVLM/ProcVLM">Code</a>
</p>

## Model Details

- **Model name:** [`ce-amtic/ProcVLM-2B`](https://huggingface.co/ce-amtic/ProcVLM-2B)
- **Model type:** Vision-language model for robot progress reward inference
- **Architecture:** Qwen3-VL-style multimodal causal language model
- **Input:** One or more RGB images sampled from a robot trajectory, plus a natural-language task description
- **Output:** Textual reasoning and a completion estimate formatted as `<progress>XX%</progress>`
- **Primary use case:** Frame-wise progress reward prediction for robotic manipulation videos

## Intended Use

ProcVLM-2B is designed for research on robot learning, progress reward modeling, embodied evaluation, and procedure-aware video understanding. Typical use cases include:

- estimating task completion progress from robot videos;
- producing dense progress rewards from sparse demonstrations;
- adapting progress prediction to a new environment with one-shot LoRA fine-tuning.

This model is not intended to be used as a safety-critical controller without downstream validation.

## Quick Start

Clone the ProcVLM repository and install the environment:

```bash
git clone https://github.com/ProcVLM/ProcVLM.git
cd ProcVLM

uv sync --python 3.10
source .venv/bin/activate
uv pip install flash-attn --no-build-isolation
```

Run progress reward inference on a video:

```bash
python evqa/inference.py \
    --model_path ce-amtic/ProcVLM-2B \
    --video_path path/to/your/video.mp4 \
    --output_path path/to/progress_predictions.jsonl \
    --task "fold the red T-shirt" \
    --window_size 8
```

Each JSONL row contains a sampled `frame_index` and its corresponding `progress` prediction.

You can also visualize predictions as a video:

```bash
python evqa/eval/visualize_progress_video.py \
    --model_path ce-amtic/ProcVLM-2B \
    --video_path path/to/your/video.mp4 \
    --output_path path/to/progress_visualization.mp4 \
    --task "fold the red T-shirt" \
    --window_size 8
```

## Python API

The same inference workflow is available through `infer_progress_from_video()`:

```python
from evqa.inference import infer_progress_from_video

records = infer_progress_from_video(
    model_path="ce-amtic/ProcVLM-2B",
    video_path="path/to/your/video.mp4",
    task="fold the red T-shirt",
    window_size=8,
)

for item in records:
    print(item["frame_index"], item["progress"])
```

The returned records include:

- `frame_index`: source video frame index;
- `timestamp_sec`: source video timestamp;
- `window_frame_indices`: frame indices used as the model input window;
- `progress`: parsed progress value in `[0, 100]`;
- `reasoning`: model reasoning with the progress tag removed;
- `model_output`: raw model output.

## Prompt Format

ProcVLM uses a procedural progress prompt. The default template is:

```text
Given the recent observation and the task "{task}", first infer the remaining atomic actions required to complete the task. Then estimate the current completion percentage and output it as a float wrapped by <progress> tags.
```

The model should answer with reasoning and a final progress tag, for example:

```text
To complete the task: Tower the blocks, the following steps are required:
1. Grasp the green block.
2. Place the green block onto the red block.
Therefore, the estimated progress percentage is <progress>84.13%</progress>.
```

Or if the task is finished:

```text
The task requires: Tower the blocks. Images show no block outside the tower, no further steps required. 
Therefore, the estimated progress percentage is <progress>100.00%</progress>.
```

## vLLM Batch Inference

For high-throughput multi-image inference, the ProcVLM repository provides `evqa.model.batch_chat_with_vllm()`:

```python
from evqa.model import batch_chat_with_vllm

outputs = batch_chat_with_vllm(
    batch_items=[
        {
            "image": [
                "frames/frame_000000.jpg",
                "frames/frame_000010.jpg",
                "frames/frame_000020.jpg",
            ],
            "conversations": [
                {
                    "from": "human",
                    "value": 'Given the recent observation and the task "fold the red T-shirt", first infer the remaining atomic actions required to complete the task. Then estimate the current completion percentage and output it as a float wrapped by <progress> tags.',
                }
            ],
        }
    ],
    model_path="ce-amtic/ProcVLM-2B",
    max_new_tokens=1024,
    temperature=0.0,
    tp=1,
)
```

## One-Shot LoRA Adaptation

ProcVLM can be adapted to a new environment with one successful task demonstration, plus optional additional successful or unsuccessful demonstrations. See the [one-shot adaptation guide](https://github.com/ProcVLM/ProcVLM/blob/main/evqa/docs/oneshot_adaptation.md) for:

- annotating coarse sub-task stages with the visual UI;
- generating a LoRA fine-tuning dataset;
- running `evqa/one-shot/lora_oneshot.sh`;
- using the saved LoRA checkpoint with `evqa/inference.py --use_lora`.

## Limitations

- The model estimates progress from visual observations and task text; it may be unreliable under strong domain shift, severe occlusion, unusual camera viewpoints, or ambiguous task descriptions.
- The progress output is a learned estimate, not a calibrated physical measurement.
- For long-horizon videos, inference quality depends on the sampled frame window and the task description.
- The model should be validated in the target robot environment before being used as a reward signal for training or deployment.

## Citation

If you use ProcVLM, please cite the paper:

```bibtex
@misc{feng2026procvlmlearningproceduregroundedprogress,
      title={ProcVLM: Learning Procedure-Grounded Progress Rewards for Robotic Manipulation}, 
      author={Youhe Feng and Hansen Shi and Haoyang Li and Xinlei Guo and Yang Wang and Chengyang Zhang and Jinkai Zhang and Xiaohan Zhang and Jie Tang and Jing Zhang},
      year={2026},
      eprint={2605.08774},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2605.08774}, 
}
```

## License

Please refer to the license information on this model repository and the upstream base model license before using the weights.
