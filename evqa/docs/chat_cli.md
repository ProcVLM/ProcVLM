# Chat CLI User Guide

This guide explains how to use `evqa/chat_cli.py`, including:
- Model loading and startup
- Real-time multi-turn conversations
- Image inputs through commands and inline tags
- Device and multi-GPU configuration with `device_map`
- Common troubleshooting tips

## 1. Overview

`chat_cli.py` is an interactive command-line tool. It loads the model through
`evqa.model.load_procvlm` and runs each inference turn with `batch_infer`.

Core capabilities:
- Supports multi-turn text conversations with local conversation history
- Supports a global image list through `/image` and `/addimage`
- Supports inline image tags in a single message, such as `[img=xxx.jpg]`
- Passes `device_map` and `torch_dtype` options through to the underlying model loader

## 2. Launching

Run from the project root:

```bash
CUDA_VISIBLE_DEVICES=0 python -m evqa.chat_cli --model-path /path/to/your/checkpoint
```

Common full example:

```bash
python -m evqa.chat_cli \
  --model-path /path/to/your/checkpoint \
  --device-map auto \
  --torch-dtype auto \
  --max-new-tokens 256 \
  --temperature 0.0
```

After startup succeeds, the interactive prompt appears:

```text
You>
```

## 3. Command-Line Arguments

- `--model-path`
  - Required. Path to the model directory.

- `--device-map`
  - Defaults to `auto`.
  - Passed to `load_procvlm(..., device_map=...)`.
  - Common values: `auto`, `cuda:0`, `cpu`.

- `--torch-dtype`
  - Defaults to `auto`.
  - Common values: `auto`, `bfloat16`, `float16`, `float32`.

- `--max-new-tokens`
  - Maximum generation length for each turn. Defaults to 256.

- `--temperature`
  - Sampling temperature. Defaults to 0.0 for more deterministic output.

- `--image`
  - Initial image path. Can be passed multiple times.
  - Example: `--image ./a.jpg --image ./b.jpg`

## 4. Interactive Commands

After entering the conversation, you can use:

- `/help`
  - Show help.

- `/quit` or `/exit`
  - Exit the program.

- `/reset`
  - Clear conversation history.

- `/image <paths...>`
  - Replace the current global image list.

- `/addimage <paths...>`
  - Append images to the current global image list.

- `/clearimage`
  - Clear the current global image list.

- `/show`
  - Show the current turn count and global image list.

Notes:
- Paths are validated and must point to files.
- Invalid paths produce warnings and are ignored.

## 5. Inline Image Tags

You can write an inline image tag directly in a normal message:

```text
Describe this image [img=./demo.jpg]
```

Multiple images are also supported:

```text
Compare the robot arm poses in these two images [img=./a.jpg] [img=./b.jpg]
```

Processing behavior:
- The program extracts all `[img=path]` tags from the input text.
- Extracted images are merged with the current global image list, with duplicates removed, and are used only for the current request.
- The `[img=...]` tags are removed from the text before the prompt is sent to the model.

Notes:
- If a path contains spaces, write it as `[img="./my image.jpg"]`.
- Invalid paths are ignored and reported with a warning.

## 6. Device Map and Multi-GPU Notes

### Recommended: limit visible GPUs first, then use `auto`

```bash
CUDA_VISIBLE_DEVICES=1,3 python -m evqa.chat_cli --model-path /path/to/ckpt --device-map auto
```

Explanation:
- Only two GPUs are visible inside the process.
- They are remapped inside the process as logical `cuda:0` and `cuda:1`.
- This is the most common and stable way to choose a subset of GPUs.

### Use a single GPU

```bash
python -m evqa.chat_cli --model-path /path/to/ckpt --device-map cuda:2
```

### CPU mode for debugging

```bash
python -m evqa.chat_cli --model-path /path/to/ckpt --device-map cpu
```

Notes:
- This tool is a single-process interactive chat tool, not a throughput-optimized parallel inference script.
- If you set `CUDA_VISIBLE_DEVICES=3` and then pass `--device-map cuda:0`, logical `cuda:0` maps to physical GPU 3.

## 7. Conversation Behavior

- The tool keeps previous turns and uses them to build the next contextual prompt.
- It is useful for Q&A, follow-up questions, and step-by-step task clarification.
- To start a new topic, run `/reset` first.

## 8. Troubleshooting

- Problem: model loading fails.
  - Check that `--model-path` is correct and the directory is complete.

- Problem: an image is reported as invalid.
  - Check that the path exists, points to a file, and is readable.

- Problem: out of memory (OOM).
  - Lower `--max-new-tokens`.
  - Use a smaller dtype, such as `--torch-dtype bfloat16` or `float16`.
  - Reduce the number or resolution of input images.

- Problem: responses are slow.
  - Reduce generation length.
  - Use a faster device or reduce conversation context by running `/reset`.

## 9. Quick Examples

Example 1: set a global image first, then ask a question.

1. Enter: `/image ./frame_001.jpg`
2. Enter: `Describe the key action in this image`

Example 2: use a temporary image for one turn.

1. Enter: `Is this part assembled correctly? [img=./part.jpg]`

Example 3: compare multiple images.

1. Enter: `Compare the differences [img=./before.jpg] [img=./after.jpg]`
