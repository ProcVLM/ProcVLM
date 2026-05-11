import concurrent.futures
import os
import numpy as np
import transformers
from PIL import Image
from copy import deepcopy
from transformers import AutoTokenizer, Qwen2VLImageProcessor
from tqdm import tqdm
from pathlib import Path
from core.utils.common import load_jsonlines, write_jsonlines, episode2num_token, hash_str
from evqa.tools.functions import iter_episode_files
from evqa.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)

def hash_num(s: str):
    try:
        return episode2num_token(s)
    except Exception as e:
        return hash_str(s)


class MultimodalProcessor:
    def __init__(self, data_args, base_processor, device='cpu'):
        self.data_args = data_args
        self.device = device
        self.image_processor = deepcopy(base_processor)
        self.image_processor.max_pixels = data_args.max_pixels
        self.image_processor.min_pixels = data_args.min_pixels
        self.image_processor.size = {
            'longest_edge': data_args.max_pixels,
            'shortest_edge': data_args.min_pixels
        }
        self.video_processor = deepcopy(base_processor)
        self.video_processor.max_pixels = data_args.video_max_pixels
        self.video_processor.min_pixels = data_args.video_min_pixels
        self.video_processor.size = {
            'longest_edge': data_args.video_max_pixels,
            'shortest_edge': data_args.video_min_pixels
        }

    def process_images(self, image_files):
        if not isinstance(image_files, list):
            image_files = [image_files]
        pil_images = []
        for image_file in image_files:
            image_path = os.path.join(self.data_args.data_path, image_file)
            if not os.path.exists(image_path):
                print(f'Image file does not exist: {image_path}')
                continue
            img = Image.open(image_path).convert('RGB')
            pil_images.append(img)
        if len(pil_images) == 0:
            return 0
        visual_processed = self.image_processor.preprocess(
            images=pil_images,
            return_tensors='pt'
        )
        # image_grid_thw shape: (B, T, H, W)
        grid = visual_processed['image_grid_thw']
        B = grid.shape[0]
        grid_flat = grid.view(B, -1)
        tokens_per_image = grid_flat.prod(dim=1)
        total_tokens = tokens_per_image.sum() // 4

        return int(total_tokens.item())

    def process_video(self, video_file):
        from torchcodec.decoders import VideoDecoder
        video_path = os.path.join(self.data_args.data_path, video_file)
        decoder = VideoDecoder(video_path, device=self.device)
        total_frames = decoder.metadata.num_frames
        avg_fps = decoder.metadata.average_fps
        video_length = total_frames / avg_fps
        interval = self.data_args.base_interval
        num_frames_to_sample = round(video_length / interval)
        target_frames = min(
            max(num_frames_to_sample, self.data_args.video_min_frames),
            self.data_args.video_max_frames
        )
        frame_idx = np.unique(
            np.linspace(0, total_frames - 1, target_frames, dtype=int)
        ).tolist()
        frame_batch = decoder.get_frames_at(indices=frame_idx)
        video_frames_numpy = frame_batch.data.cpu().numpy()
        visual_processed = self.video_processor.preprocess(
            images=None,
            videos=video_frames_numpy,
            return_tensors='pt'
        )
        grid = visual_processed['video_grid_thw']
        total_tokens = grid.prod(dim=1).sum() // 4
        return int(total_tokens.item())
    

def calculate_tokens(conversation, processor, tokenizer):
    total_tokens = 21
    roles = {'human': 'user', 'gpt': 'assistant'}
    for message in conversation['conversations']:
        role = message['from']
        text = message['value']
        conv = [{'role': roles[role], 'content': text}]
        encode_id = tokenizer.apply_chat_template(conv, return_tensors='pt', add_generation_prompt=False)[0]
        total_tokens += len(encode_id)
    if 'image' in conversation:
        images = conversation['image'] if isinstance(conversation['image'], list) else [conversation['image']]
        total_tokens += processor.process_images(images)
    elif 'video' in conversation:
        videos = conversation['video'] if isinstance(conversation['video'], list) else [conversation['video']]
        for video_file in videos:
            total_tokens += processor.process_video(video_file)
    return total_tokens


def load_processor_and_tokenizer(model_path, data_args):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    base_image_processor = Qwen2VLImageProcessor.from_pretrained(model_path)
    print(f'Successfully loaded model components from {model_path}')
    processor = MultimodalProcessor(data_args, base_image_processor, device='cpu')
    return processor, tokenizer


def calculate_and_update(item):
    item['num_tokens'] = int(calculate_tokens(item, processor, tokenizer))
    return item


# --- Main Execution ---
if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    parser.add_argument("--json_root", type=str, help="Root directory for RAW organised JSON files.", required=True)
    parser.add_argument("--data_root", type=str, help="Root directory for data files (images/videos).", required=True)
    parser.add_argument("--task_ids", type=str, nargs='+', help="List of task IDs to process.", choices=["a", "b", "c"], default=["a", "b", "c"])
    parser.add_argument("--dataset_paths", type=str, nargs='+', help="List of dataset paths to process.")
    parser.add_argument("--force_recompute", action='store_true', help="Whether to force recompute token counts even if they exist.")
    parser.add_argument("--skip_processing", action='store_true', help="Whether to skip processing and just print stats.")
    parser.add_argument("--show_QA_counts_per_task", action='store_true', help="Whether to show QA pair counts per task.")
    parser.add_argument("--search_for_suffix", type=str, default=None, help="Suffix to search for if resize is applied to images/videos.")
    parser.add_argument("--search_for_long_request_above", type=int, default=None, help="Search for requests with length above this value.")
    parser.add_argument("--world_size", type=int, default=1, help="Total number of parallel processes.")
    parser.add_argument("--rank", type=int, default=0, help="Rank of the current process.")
    model_args, data_args, training_args, args = parser.parse_args_into_dataclasses()
    
    print(f'Image configuration: max_pixels={data_args.max_pixels}, min_pixels={data_args.min_pixels}')
    print(f'Video frame configuration: video_max_pixels={data_args.video_max_pixels}, video_min_pixels={data_args.video_min_pixels}')

    dataset_names = None
    if args.dataset_paths:
        dataset_names = [Path(p).name for p in args.dataset_paths]
    
    if not args.skip_processing:
        model_path = model_args.model_name_or_path
        processor, tokenizer = load_processor_and_tokenizer(model_path, data_args)
        processor.data_args.data_path = args.data_root
        suffix = args.search_for_suffix

        num_tokens_stats = {
            task_id: {} for task_id in args.task_ids
        }
        num_qa_pairs_stats = {
            task_id: 0 for task_id in args.task_ids
        }
        if args.search_for_long_request_above is not None:
            long_request_stats = {
                task_id: {} for task_id in args.task_ids
            }
        for cluster, ds_name, ep in tqdm(iter_episode_files(args.json_root, cluster_filter=args.task_ids), desc=f"Counting Tokens {args.task_ids}"):
            if dataset_names and ds_name not in dataset_names:
                continue
            if hash_num(str(ep)) % args.world_size != args.rank:
                continue
            ep_data = load_jsonlines(ep)
            # try to skip already computed ones
            recompute_needed = args.force_recompute
            if not recompute_needed and not all('num_tokens' in item for item in ep_data):
                recompute_needed = True
            if not recompute_needed and suffix: # special check for resized images/videos
                for item in ep_data:
                    if 'image' in item:
                        images = item['image'] if isinstance(item['image'], list) else [item['image']]
                        for image_file in images:
                            # images/Table30_open_the_drawer/observation.images.global_image/episode_000012/000000_320.jpg
                            image_stem = image_file.split('.')[-2] # .../000000_320
                            if image_stem.endswith(suffix):
                                recompute_needed = True
                                break
                    elif 'video' in item: # only single video case considered here
                        video_file = item['video']
                        video_stem = video_file.split('.')[-2]
                        if video_stem.endswith(suffix):
                            recompute_needed = True
                    if recompute_needed:
                        break
            # skip if all have num_tokens and no recompute needed
            if not recompute_needed:
                if ds_name not in num_tokens_stats[cluster]:
                    num_tokens_stats[cluster][ds_name] = 0
                num_tokens_stats[cluster][ds_name] += sum([item['num_tokens'] for item in ep_data])
                num_qa_pairs_stats[cluster] += len(ep_data)
                if args.search_for_long_request_above is not None:
                    for item in ep_data:
                        if item['num_tokens'] > args.search_for_long_request_above:
                            if ds_name not in long_request_stats[cluster]:
                                long_request_stats[cluster][ds_name] = (0, 0)
                            long_request_stats[cluster][ds_name] = (
                                long_request_stats[cluster][ds_name][0] + 1,
                                long_request_stats[cluster][ds_name][1] + item['num_tokens']
                            )
                continue

            # --- modify data path in processor ---
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                ep_data_with_tokens = list(executor.map(calculate_and_update, ep_data))
            write_jsonlines(ep_data_with_tokens, ep)

            if ds_name not in num_tokens_stats[cluster]:
                num_tokens_stats[cluster][ds_name] = 0
            num_tokens_stats[cluster][ds_name] += sum([item['num_tokens'] for item in ep_data_with_tokens])
            num_qa_pairs_stats[cluster] += len(ep_data_with_tokens)
            if args.search_for_long_request_above is not None:
                for item in ep_data_with_tokens:
                    if item['num_tokens'] > args.search_for_long_request_above:
                        if ds_name not in long_request_stats[cluster]:
                            long_request_stats[cluster][ds_name] = (0, 0)
                        long_request_stats[cluster][ds_name] = (
                            long_request_stats[cluster][ds_name][0] + 1,
                            long_request_stats[cluster][ds_name][1] + item['num_tokens']
                        )
    else:
        num_tokens_stats = {
            task_id: {} for task_id in args.task_ids
        }
        num_qa_pairs_stats = {
            task_id: 0 for task_id in args.task_ids
        }
        if args.search_for_long_request_above is not None:
            long_request_stats = {
                task_id: {} for task_id in args.task_ids
            }
        for cluster, ds_name, ep in tqdm(iter_episode_files(args.json_root, cluster_filter=args.task_ids), desc=f"Loading Token Stats {args.task_ids}"):
            if dataset_names and ds_name not in dataset_names:
                continue
            ep_data = load_jsonlines(ep)
            if all('num_tokens' in item for item in ep_data):
                if ds_name not in num_tokens_stats[cluster]:
                    num_tokens_stats[cluster][ds_name] = 0
                num_tokens_stats[cluster][ds_name] += sum([item['num_tokens'] for item in ep_data])
                num_qa_pairs_stats[cluster] += len(ep_data)
                if args.search_for_long_request_above is not None:
                    for item in ep_data:
                        if item['num_tokens'] > args.search_for_long_request_above:
                            if ds_name not in long_request_stats[cluster]:
                                long_request_stats[cluster][ds_name] = (0, 0)
                            long_request_stats[cluster][ds_name] = (
                                long_request_stats[cluster][ds_name][0] + 1,
                                long_request_stats[cluster][ds_name][1] + item['num_tokens']
                            )
            else:
                print(f'Warning: num_tokens not found in {ep}, skipping.')

    if args.world_size == 1:
        print("Token count statistics per task and dataset:")
        in_total = 0
        for task_id in args.task_ids:
            ds_stats = num_tokens_stats[task_id]
            task_num = sum(ds_stats.values())
            in_total += task_num
            print(f"{task_id}: {task_num/1e9:.3f} B")
            for ds_name, total_tokens in ds_stats.items():
                print(f"  {ds_name}: {total_tokens/1e9:.3f} B")
        print(f"Overall total: {in_total/1e9:.3f} B")

        if args.search_for_long_request_above is not None:
            print(f"Requests with tokens above {args.search_for_long_request_above}:")
            long_request_in_total, long_request_tokens_in_total = 0, 0
            for task_id in args.task_ids:
                ds_stats = long_request_stats[task_id]
                task_num, task_tokens = 0, 0
                for ds_name, (count, total_tokens) in ds_stats.items():
                    task_num += count
                    task_tokens += total_tokens
                long_request_in_total += task_num
                long_request_tokens_in_total += task_tokens
                print(f"{task_id}: {task_num} requests")
                for ds_name, (count, total_tokens) in ds_stats.items():
                    print(f"  {ds_name}: {count} requests (Total tokens: {total_tokens/1e9:.3f} B)")
            print(f"Overall total requests above {args.search_for_long_request_above}: {long_request_in_total}")
            print(f"Overall total tokens for requests above {args.search_for_long_request_above}: {long_request_tokens_in_total/1e9:.3f} B")

        qa_pair_in_total = 0
        for task_id in args.task_ids:
            task_num = num_qa_pairs_stats[task_id]
            qa_pair_in_total += task_num
            if args.show_QA_counts_per_task:
                print(f"{task_id}: {task_num} QA pairs")
        print(f"Overall total QA pairs: {qa_pair_in_total}")
