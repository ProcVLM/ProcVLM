import os
import argparse
import json
import math
import ast
from pathlib import Path
from functools import partial
from datetime import datetime
from tqdm import tqdm
from evqa.data import data_list
from evqa.eval.compute_metrics import (
    compute_accuracy_with_episode,
    extract_episode_token,
    auto_detect_task_type,
)

################## Define format prompts for different task types ############################
#   These prompts are not use in the SFT process, or the evaluation of trained models,       #
#     as the output format is already known to the model.                                    #
#   But to achieve a fair comparison across models that are not specifically fine-tuned      #
#     for certain tasks, we append these format prompts to the questions during evaluation.  #
##############################################################################################
format_prompts = {
    'segmentation': (
        " Please answer with a list of predicted segments, each segment formatted as a dictionary with required keys."
        "e.g. [{<start_key_name>: 0, <end_key_name>: 10, <label_key_name>: 'label1'},"
        "      {<start_key_name>: 11, <end_key_name>: 20, <label_key_name>: 'label2'}, ...]"
    ),
}
###############################################################################################


### Define command-line arguments ###
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model checkpoint.")
    parser.add_argument("--dataset_use", type=str, default=None, help="Dataset names separated by comma.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation.")
    parser.add_argument("--model_type", type=str, default=None, choices=['qwen', 'intern', 'procvlm', 'api'])
    parser.add_argument("--log_dir", type=str, default="./logs/eval", help="Directory to save evaluation results.")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism degree.")
    parser.add_argument("--recover_from_log", type=str, default=None, help="Recover predictions from a previous eval log.")
    parser.add_argument("--enable_value_head", action="store_true", default=False, help="Enable value head for procvlm even if model supports it.")
    return parser.parse_args()


### Mode 1: Run inference evaluation with model and compute metrics, then log predictions and results to file ###
def run_inference_eval(args):
    if not args.dataset_use:
        raise ValueError("--dataset_use is required when not in recover mode.")

    model_type, batch_generate = build_batch_generator(args)

    os.makedirs(args.log_dir, exist_ok=True)
    model_token = Path(args.model_path).name.replace(".", "_")
    if args.enable_value_head:
        model_token += "_with_vh"
    logging_file = os.path.join(args.log_dir, f"results_{model_type}_{model_token}.log")
    if os.path.exists(logging_file):
        os.remove(logging_file)

    dataset_names = [name.strip() for name in args.dataset_use.split(",") if name.strip()]
    configs = data_list(dataset_names)
    for data_name, config in zip(dataset_names, configs):
        print(f"\n=======================================================")
        print(f"Evaluating on dataset: {data_name}")
        print(f"Annotation Path: {config['annotation_path']}")
        print(f"Data Path: {config['data_path']}")
        if 'sampling_rate' in config:
            print(f"[WARNING] Sampling rate {config['sampling_rate']} is ignored in eval.")
        print(f"=======================================================")
        append_to_logfile(logging_file, f"===== Evaluation Results for Dataset: {data_name} =====")
        append_to_logfile(logging_file, f"Annotation Path: {config['annotation_path']}")
        append_to_logfile(logging_file, f"Data Path: {config['data_path']}\n")

        anno_path = config['annotation_path']
        data_root = config['data_path']

        data_items = []
        with open(anno_path, 'r') as f:
            for line in tqdm(f, desc="Loading data"):
                if line.strip():
                    item = json.loads(line)
                    item = make_abs_path(item, data_root)
                    data_items.append(item)
        print(f"Loaded {len(data_items)} samples.")

        all_preds = []
        all_gts = []
        episode_tokens = []

        if args.batch_size <= 0:
            args.batch_size = len(data_items) # let the model handle all samples in one batch if batch_size <= 0
            
        num_batches = math.ceil(len(data_items) / args.batch_size)
        for i in tqdm(range(num_batches), desc=f"Inferencing {data_name}"):
            start_idx = i * args.batch_size
            end_idx = min((i + 1) * args.batch_size, len(data_items))
            batch_items = data_items[start_idx:end_idx]
            for item in batch_items:
                gt = item["conversations"][1]["value"]
                cate = auto_detect_task_type(gt)
                if cate in format_prompts:
                    prompt_suffix = format_prompts[cate]
                    item["conversations"][0]["value"] += prompt_suffix
                epi = extract_episode_token(item)
                episode_tokens.append(epi)

            batch_gts = [item["conversations"][1]["value"] for item in batch_items]
            all_gts.extend(batch_gts)
            for item in batch_items:
                if len(item["conversations"]) > 1:
                    item["conversations"] = item["conversations"][:1]

            batch_preds = batch_generate(batch_items)
            all_preds.extend(batch_preds)
            for item, gt, pred in zip(batch_items, batch_gts, batch_preds):
                log_content = f"----------\nResources: {item.get('image') or item.get('video') or None}\nQuestion: {item['conversations'][0]['value']}\nGround Truth: {gt}\nPrediction: {pred}\n"
                append_to_logfile(logging_file, log_content)

        print(f"Computing metrics for {data_name}...")
        assert len(all_preds) == len(all_gts), "Number of predictions and ground truths must match."
        metrics = compute_accuracy_with_episode(
            pred_txts=all_preds,
            gt_txts=all_gts,
            episodes=episode_tokens,
            include_deprecated=False,
        )

        print(f"===== Evaluation Results for Dataset: {data_name} =====")
        append_to_logfile(logging_file, f"===== Evaluation Results for Dataset: {data_name} =====")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
            append_to_logfile(logging_file, f"{metric_name}: {metric_value:.4f}")
        print(f"Results logged to {logging_file}")

    return 0


### Mode 2: Recover predictions from existing log file and re-compute metrics ###
def run_recover_from_log(args):
    if not os.path.exists(args.recover_from_log):
        raise FileNotFoundError(f"Recover log file not found: {args.recover_from_log}")

    print(f"Recovering predictions from log: {args.recover_from_log}")
    dataset_logs, dataset_orig_metrics = parse_recover_log(args.recover_from_log)

    if args.dataset_use:
        dataset_names = [name.strip() for name in args.dataset_use.split(",") if name.strip()]
        print(f"Using user-specified datasets in recover mode: {dataset_names}")
    else:
        dataset_names = list(dataset_logs.keys())
        print(f"Auto-detected datasets from recover log: {dataset_names}")

    if len(dataset_names) == 0:
        raise ValueError("No datasets found for recovery. Please check --recover_from_log or provide --dataset_use.")

    configs = data_list(dataset_names)

    os.makedirs(args.log_dir, exist_ok=True)
    recover_token = Path(args.recover_from_log).stem.replace("results_", "recover_")
    logging_file = os.path.join(args.log_dir, f"{recover_token}.log")
    if os.path.exists(logging_file):
        os.remove(logging_file)

    for data_name, config in zip(dataset_names, configs):
        print(f"\n=======================================================")
        print(f"Recover metrics on dataset: {data_name}")
        print(f"Annotation Path: {config['annotation_path']}")
        print(f"Data Path: {config['data_path']}")
        print(f"=======================================================")
        append_to_logfile(logging_file, f"===== Recovery Results for Dataset: {data_name} =====")
        append_to_logfile(logging_file, f"Annotation Path: {config['annotation_path']}")
        append_to_logfile(logging_file, f"Data Path: {config['data_path']}\n")

        if data_name not in dataset_logs:
            print(f"[WARNING] Dataset {data_name} not found in recover log, skip.")
            append_to_logfile(logging_file, f"[WARNING] Dataset {data_name} not found in recover log, skip.\n")
            continue

        recovered_samples = dataset_logs[data_name]
        all_preds = []
        all_gts = []
        episode_tokens = []

        for sample in recovered_samples:
            gt = sample.get("gt", "")
            pred = sample.get("pred", "")
            question = sample.get("question", "")
            resources = sample.get("resources", None)

            item = {
                "conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": gt},
                ]
            }
            if resources is not None:
                if isinstance(resources, list):
                    if len(resources) > 0:
                        first = str(resources[0]).lower()
                        if any(first.endswith(ext) for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]):
                            item["video"] = resources
                        else:
                            item["image"] = resources
                else:
                    item["image"] = [resources]

            epi = extract_episode_token(item)
            episode_tokens.append(epi)
            all_gts.append(gt)
            all_preds.append(pred)

        print(f"Recovered {len(all_preds)} valid samples from log for {data_name}.")
        append_to_logfile(logging_file, f"Recovered Samples: {len(all_preds)}")

        if len(all_preds) == 0:
            print(f"[WARNING] No valid recovered samples for dataset {data_name}, skip.")
            append_to_logfile(logging_file, f"[WARNING] No valid recovered samples for dataset {data_name}, skip.\n")
            continue

        metrics = compute_accuracy_with_episode(
            pred_txts=all_preds,
            gt_txts=all_gts,
            episodes=episode_tokens,
            include_deprecated=False,
        )

        print(f"===== Recovery Results for Dataset: {data_name} =====")
        append_to_logfile(logging_file, f"===== Recovery Results for Dataset: {data_name} =====")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
            append_to_logfile(logging_file, f"{metric_name}: {metric_value:.4f}")

        target_metrics = ["EPR@50", "VOC"]
        orig_metrics = dataset_orig_metrics.get(data_name, {})
        mismatch_warnings = []
        for tm in target_metrics:
            if tm in orig_metrics and tm in metrics:
                if abs(metrics[tm] - orig_metrics[tm]) > 1e-3:
                    mismatch_warnings.append(f"Metric {tm} changed significantly! Original: {orig_metrics[tm]:.4f}, Recovered: {metrics[tm]:.4f}")

        if mismatch_warnings:
            print("\n[WARNING] Metric Verification Failed:")
            append_to_logfile(logging_file, "[WARNING] Metric Verification Failed:")
            for w in mismatch_warnings:
                print(w)
                append_to_logfile(logging_file, w)
        else:
            success_msg = "[INFO] Metric Verification Passed."
            print(success_msg)
            append_to_logfile(logging_file, success_msg)

    print(f"\nRecovery results logged to {logging_file}")
    return 0


### Helper Functions ###
def make_abs_path(item, data_root):
    """In-place update of image/video paths in item to be absolute paths."""
    if "image" in item:
        images = item["image"]
        if isinstance(images, str):
            item["image"] = [os.path.join(data_root, images)]
        elif isinstance(images, list):
            item["image"] = [os.path.join(data_root, img) for img in images]
    if "video" in item:
        videos = item["video"]
        if isinstance(videos, str):
            item["video"] = [os.path.join(data_root, videos)]
        elif isinstance(videos, list):
            item["video"] = [os.path.join(data_root, vid) for vid in videos]
    return item

def append_to_logfile(logfile, content):
    with open(logfile, 'a') as f:
        f.write(content + '\n')

def build_batch_generator(args):
    if not args.model_type:
        raise ValueError("Model type must be specified with --model_type when not in recover mode.")
    if not args.model_path:
        raise ValueError("--model_path is required when not in recover mode.")

    model_type = args.model_type
    print(f"Using Model Type: {model_type}")

    if model_type == "intern":
        from core.models.internvl import batch_chat_with_lmd
        batch_generate = partial(
            batch_chat_with_lmd, 
            tp=args.tp, 
            model_path=args.model_path
        )
    elif model_type == "procvlm":
        if args.enable_value_head:
            from evqa.model import batch_chat_with_value_head
            if args.tp > 1:
                print(f"[INFO] procvlm uses data parallel batch splitting on {args.tp} visible GPUs.")
            batch_generate = partial(
                batch_chat_with_value_head, 
                model_path=args.model_path,
                dp=args.tp, 
                torch_dtype="bf16", 
                image_cost_mb=250, 
                enable_value_head=True
            )
        else:
            from evqa.model import batch_chat_with_vllm
            engine_kwargs = {
                "gpu_memory_utilization": 0.9,
                "max_model_len": 32768,
            }
            sampling_kwargs = {
                "temperature": 0.0
            }
            batch_generate = partial(
                batch_chat_with_vllm,
                model_path=args.model_path,
                tp=args.tp,
                max_new_tokens=8192,
                engine_kwargs=engine_kwargs,
                sampling_kwargs=sampling_kwargs,
            )
    elif model_type == "qwen":
        from core.models.qwenvl import batch_chat_with_vllm
        qwenvl_kwargs = { }
        qwen_sampling_kwargs = {"temperature": 0.0}
        if "235b" in args.model_path.lower():
            qwenvl_kwargs = {
                "quantization": "fp8",
                "gpu_memory_utilization": 0.85,
                "enable_expert_parallel": True,
                "max_model_len": 32768,
                "max_num_seqs": 128,
                "distributed_executor_backend": "mp",
                "mm_encoder_tp_mode": "data",
                "disable_custom_all_reduce": True,
            }
        elif "27b" in args.model_path.lower():
            qwenvl_kwargs = {
                "gpu_memory_utilization": 0.7,
                "max_model_len": 81920,
            }
            qwen_sampling_kwargs = {
                "temperature": 1.0,
                "top_p": 0.95,
                "presence_penalty": 1.5,
                "top_k": 20,
            }
        batch_generate = partial(
            batch_chat_with_vllm,
            model_path=args.model_path,
            tp=args.tp,
            max_tokens=81920,
            engine_kwargs=qwenvl_kwargs,
            sampling_kwargs=qwen_sampling_kwargs,
        )
    elif model_type == "api":
            from core.backends.api_v2 import batch_generate as api_batch_generate
            
            model_name_lower = args.model_path.lower()
            if "gemini" in model_name_lower:
                api_namespace = "GEMINI_API_KEY"
            else:
                api_namespace = "OPENAI_API_KEY"
                
            batch_generate = partial(
                api_batch_generate,
                model_path=args.model_path,
                base_url="...",
                api_key_namespace=api_namespace,
                max_workers=min(args.batch_size, 32)
            )
    else:
        raise ValueError(f"Unsupported model type: {model_type}, please modify the code to add support.")

    return model_type, batch_generate

def parse_recover_log(log_path):
    """Parse an existing eval log file and recover samples grouped by dataset."""
    dataset_blocks = {}
    dataset_original_metrics = {}
    current_dataset = None
    current_sample = None
    current_field = None

    def finalize_current_sample():
        nonlocal current_sample, current_field
        if current_dataset is None or current_sample is None:
            current_sample = None
            current_field = None
            return

        for key in ["question", "gt", "pred"]:
            if current_sample.get(key) is not None:
                current_sample[key] = current_sample[key].rstrip("\n")

        if any([
            current_sample.get("question", "") != "",
            current_sample.get("gt", "") != "",
            current_sample.get("pred", "") != "",
            current_sample.get("resources", None) is not None,
        ]):
            dataset_blocks.setdefault(current_dataset, []).append(current_sample)

        current_sample = None
        current_field = None

    with open(log_path, "r") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if line.startswith("===== Evaluation Results for Dataset:"):
                finalize_current_sample()
                suffix = line[len("===== Evaluation Results for Dataset:"):].strip()
                if suffix.endswith("====="):
                    suffix = suffix[:-5].rstrip()
                current_dataset = suffix
                dataset_blocks.setdefault(current_dataset, [])
                continue

            if current_dataset is None:
                continue

            if line.strip() == "----------":
                finalize_current_sample()
                current_sample = {
                    "resources": None,
                    "question": "",
                    "gt": "",
                    "pred": "",
                }
                current_field = None
                continue

            if current_sample is None:
                if ":" in line:
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        k = parts[0].strip()
                        v = parts[1].strip()
                        try:
                            dataset_original_metrics.setdefault(current_dataset, {})[k] = float(v)
                        except ValueError:
                            pass
                continue

            if line.startswith("Resources:"):
                resource_txt = line[len("Resources:"):].strip()
                try:
                    current_sample["resources"] = ast.literal_eval(resource_txt)
                except Exception:
                    current_sample["resources"] = resource_txt
                current_field = "resources"
            elif line.startswith("Question:"):
                current_sample["question"] = line[len("Question:"):].lstrip()
                current_field = "question"
            elif line.startswith("Ground Truth:"):
                current_sample["gt"] = line[len("Ground Truth:"):].lstrip()
                current_field = "gt"
            elif line.startswith("Prediction:"):
                current_sample["pred"] = line[len("Prediction:"):].lstrip()
                current_field = "pred"
            else:
                if current_field in ["question", "gt", "pred"]:
                    if current_sample[current_field] == "":
                        current_sample[current_field] = line
                    else:
                        current_sample[current_field] += "\n" + line
                elif current_field == "resources":
                    current_sample["resources"] = f"{current_sample['resources']}\n{line}".strip()

    finalize_current_sample()
    return dataset_blocks, dataset_original_metrics


### Evaluation entry point ###
def main():
    args = parse_args()
    if args.recover_from_log is not None:
        return run_recover_from_log(args)
    return run_inference_eval(args)

if __name__ == "__main__":
    raise SystemExit(main())


