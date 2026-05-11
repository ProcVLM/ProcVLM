import logging
import time
import multiprocessing as mp
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from core.utils.common import (
    load_jsonlines,
    write_jsonlines
)
from core.data.reader import LeRobotDatasetMetadata 
from typing import List, Tuple, Optional, Dict, Any, Union

### DP Helpers ###
def auto_split_datasets(dataset_paths: List[str], world_size: int, rank: Optional[int] = None) -> Tuple[List[str], int, int]:
    """
    Automatically split datasets among multiple workers to balance the total number of frames.
    Would smartly decide frame-wise or dataset-wise splitting based on the number of frames and workers.
    If a rank is assigned frame-wise splitting, returns ([one target dataset path], total splits, split rank).
    Otherwise, returns (list of dataset paths for this rank, 1, 0).
    The output would be steady for same input lists, regardless of the order of input dataset paths.

    Args:
        dataset_paths: List of dataset paths to be split.
        world_size: Total number of workers.
        rank: If specified, return only the split for this rank. Otherwise, return all splits.

    Returns:
        If rank is specified, returns a tuple (split_for_rank, total_splits, split_rank).
        If rank is None, returns a list of splits for all ranks.
    """
    datasets = []
    for dp in dataset_paths:
        dm = LeRobotDatasetMetadata(repo_id=dp, root=dp)
        datasets.append((dm.total_frames, dp))
    datasets = sorted(datasets, key=lambda x: (-x[0], x[1]))
    expected = sum([d[0] for d in datasets]) / world_size
    
    splits = [([], 1, 0) for _ in range(world_size)]
    split_lens = [0 for _ in range(world_size)]
    
    i, alloc_i = 0, 0
    while i < len(datasets) and datasets[i][0] >= expected:
        partial_ws = round(datasets[i][0] / expected)
        if alloc_i + partial_ws >= world_size and i != len(datasets) - 1:
            break
        for partial_rk in range(partial_ws):
            splits[alloc_i] = ([datasets[i][1]], partial_ws, partial_rk)
            split_lens[alloc_i] = round(datasets[i][0] / partial_ws)
            alloc_i += 1
        i += 1
    
    while i < len(datasets):
        d = datasets[i]
        # find leftmost argmax(rest_space - d[0])
        maxp, argp = -float('inf'), -1
        for j in range(alloc_i, world_size):
            rest_space = expected - split_lens[j]
            if rest_space - d[0] > maxp:
                maxp = rest_space - d[0]
                argp = j
        splits[argp][0].append(d[1])
        split_lens[argp] += d[0]
        i += 1
    
    assert all([len(s[0]) > 0 for s in splits]), \
        (f"Some splits are empty. This means world_size ({world_size}) is too large for the datasets. Try a smaller one!\n"
        "Splits: {splits}")
    max_diff = (max(split_lens) - min(split_lens)) / min(split_lens)
    logging.info(f"Dataset split among {world_size} workers with maximum length difference {max_diff:.2%}")
    logging.debug(f"{split_lens}")

    if rank is None:
        return splits
    return splits[rank]


def auto_split_datasets_legacy(dataset_paths: str, world_size: int, rank: Union[int, None] = None) -> List[str]:
    assert len(dataset_paths) >= world_size, f"Number of datasets ({len(dataset_paths)}) must be at least world_size ({world_size})"
    datasets = []
    for dp in dataset_paths:
        dm = LeRobotDatasetMetadata(repo_id=dp, root=dp)
        datasets.append((dm.total_frames, dp))
    datasets = sorted(datasets, key=lambda x: (-x[0], x[1]))
    if len(datasets) == world_size:
        datasets = [d[1] for d in datasets]
        return datasets[rank]
    
    expected = sum([d[0] for d in datasets]) / world_size
    splits = [[] for _ in range(world_size)]
    split_lens = [0 for _ in range(world_size)]
    current_split_index = 0
    last_i = 0
    for i in range(len(datasets)) :
        if datasets[i][0] < expected:
            last_i = i
            break
        splits[current_split_index].append(datasets[i][1])
        split_lens[current_split_index] += datasets[i][0]
        current_split_index += 1

    datasets = datasets[last_i:]
    for d in datasets:
        # find leftmost argmax|rest_space - d[0]|
        maxp, argp = -float('inf'), -1
        for i in range(current_split_index, world_size):
            rest_space = expected - split_lens[i]
            if rest_space - d[0] > maxp:
                maxp = rest_space - d[0]
                argp = i
        splits[argp].append(d[1])
        split_lens[argp] += d[0]
    
    assert all([len(s) > 0 for s in splits]), f"Some splits are empty: {splits}"
    max_diff = (max(split_lens) - min(split_lens)) / min(split_lens)
    logging.info(f"Dataset split among {world_size} workers with maximum length difference  {max_diff:.2%}")
    
    if rank is None:
        return splits
    return splits[rank]


def auto_split_datasets_entirely(dataset_paths: List[str], world_size: int, rank: Optional[int] = None) -> Union[List[List[str]], List[str]]:
    """
    Automatically split datasets among multiple workers to balance the total number of frames.
    Unlike the frame-wise split, this variant ensures that a single dataset is assigned 
    entirely to one bucket (worker). A dataset is never split across multiple workers.
    
    The output would be steady for the same input lists, regardless of the order of input dataset paths.

    Args:
        dataset_paths: List of dataset paths to be split.
        world_size: Total number of workers.
        rank: If specified, return only the list of dataset paths for this rank. 
              Otherwise, return all splits.

    Returns:
        If rank is specified, returns a list of dataset paths: List[str].
        If rank is None, returns a list of lists of dataset paths: List[List[str]].
    """
    datasets = []
    for dp in dataset_paths:
        dm = LeRobotDatasetMetadata(repo_id=dp, root=dp)
        datasets.append((dm.total_frames, dp))
        
    # Sort descending by frames first, then alphabetically by path to guarantee deterministic output
    datasets = sorted(datasets, key=lambda x: (-x[0], x[1]))
    
    # Edge Case Protection: If we can't split datasets, world_size cannot exceed dataset count
    if world_size > len(datasets):
        raise ValueError(
            f"Cannot assign {len(datasets)} entire datasets to {world_size} workers without leaving "
            "some workers empty. Please reduce world_size or use frame-wise splitting."
        )

    splits = [[] for _ in range(world_size)]
    split_lens = [0 for _ in range(world_size)]
    
    # Greedy allocation: assign the largest unassigned dataset to the bucket with the minimum load
    for frames, dp in datasets:
        # Find the bucket with the minimum current length. 
        # min() will naturally pick the smaller index in case of a tie, ensuring stable output.
        min_idx = min(range(world_size), key=lambda j: split_lens[j])
        splits[min_idx].append(dp)
        split_lens[min_idx] += frames
    
    # Sanity check
    assert all([len(s) > 0 for s in splits]), \
        f"Some splits are empty. Try a smaller world_size!\nSplits: {splits}"
        
    max_diff = (max(split_lens) - min(split_lens)) / max(1, min(split_lens))
    logging.info(f"Dataset entirely split among {world_size} workers with maximum length difference {max_diff:.2%}")
    logging.debug(f"Split lengths: {split_lens}")

    if rank is None:
        return splits
    return splits[rank]
            

### Seperate Parts for Runner Pipeline ###
# The runner consists of a producer, maybe a preprocessor, a GPU worker, and a consumer, to form a pipeline and maximize the throughput.
def process_task(input_queue: mp.Queue, output_queue: mp.Queue, process_func: callable, batch_size: int, refresh_freq: int):
    """
    This function defines a processor process that reads items from an input queue,
    applies a processing function, and writes results to an output queue.

    Args:
        input_queue: multiprocessing.Queue from which to read items
        output_queue: multiprocessing.Queue to which to write processed items
        process_func: a callable that takes an item and returns a processed item
        refresh_freq: frequency (in number of processed items) to log speed information

    Each item in input_queue is (image, question, _supp) or None (end signal)
        image is supposed to be a PIL image or a list of PIL images
        question is a string
        _supp is a placeholder for any supplementary data, not used in processing
    
    Each item in output_queue is (image, llm_inputs_batch, _supp) or None (end signal)
        image is the same as input
        llm_inputs_batch is the output from process_func
        _supp is the same as input
    """

    start_time = last_time = time.time()
    total_count = 0
    batch: List[Tuple[Image.Image, str, Any]] = []

    def flush_batch():
        nonlocal total_count, last_time
        if not batch:
            return
        imgs, qs, supps = zip(*batch)
        results = process_func(list(imgs), list(qs))
        for img, res, supp in zip(imgs, results, supps):
            output_queue.put((img, res, supp))
            total_count += 1
            if total_count % refresh_freq == 0:
                elapsed = time.time() - last_time
                speed = refresh_freq / elapsed if elapsed > 0 else float('inf')
                speed_mtpd = speed * 86400 / 1e6
                logging.info(
                    f"Processor throughput: {speed:.2f} it/s ({speed_mtpd:.2f} mits/d). "
                    f"Processed {total_count} items."
                )
                last_time = time.time()
        batch.clear()

    while True:
        item = input_queue.get()
        if item is None:
            # Flush remaining items, signal end, log summary
            flush_batch()
            output_queue.put(None)
            elapsed = time.time() - start_time
            final_speed = total_count / elapsed if elapsed > 0 else float('inf')
            speed_mtpd = final_speed * 86400 / 1e6
            logging.info(f"Processor done: processed {total_count} items, avg throughput is {final_speed:.2f} it/s ({speed_mtpd:.2f} mits/d) over {elapsed:.2f}s.")
            break
        batch.append(item)
        if len(batch) >= batch_size:
            flush_batch()

def gpu_worker(input_queue: mp.Queue, output_queue: mp.Queue, infer_func: callable, batch_size: int, refresh_freq: int):
    """
    This function defines a GPU working process that performs inference on batches of images and questions.
    It reads from an input queue and writes results to an output queue.

    Args:
        input_queue: multiprocessing.Queue from which to read (image, question) pairs
        output_queue: multiprocessing.Queue to which to write (image, question, generated_text) triples
        infer_func: a callable that takes a list of images and a list of questions and returns a list of generated texts
        batch_size: number of (image, question) pairs to process in one batch
        refresh_freq: frequency (in number of processed items) to log speed information

    Each item in input_queue is (image, question, _supp) or None (end signal)
        image is supposed to be a PIL image or a list of PIL images
        question is a string
        _supp is a placeholder for any supplementary data, not used in inference

    Each item in output_queue is (image, question, generated_text, _supp) or None (end signal)
        image, question, _supp is the same as input
        generated_text is the output from infer_func
    """
    def flush_batch():
        if not batch:
            return
        imgs, qs, supps = zip(*batch)
        results = infer_func(list(imgs), list(qs))
        for img, q, res, supp in zip(imgs, qs, results, supps):
            output_queue.put((img, q, res, supp))
        batch.clear()

    start_time = time.time()
    batch: List[Tuple[Image.Image, str, Any]] = []
    total_count = 0
    while True:
        item = input_queue.get()
        if item is None:
            # Flush remaining items, signal end, log summary
            flush_batch()
            output_queue.put(None)
            elapsed = time.time() - start_time
            final_speed = total_count / elapsed if elapsed > 0 else float('inf')
            speed_mtpd = final_speed * 86400 / 1e6
            logging.info(f"GPU worker done: processed {total_count} items, avg throughput is {final_speed:.2f} it/s ({speed_mtpd:.2f} mits/d) over {elapsed:.2f}s.")
            break
        batch.append(item)
        if len(batch) >= batch_size:
            flush_batch()
        total_count += 1
        if total_count % refresh_freq == 0:
            elapsed = time.time() - start_time
            speed = total_count / elapsed if elapsed > 0 else float('inf')
            speed_mtpd = speed * 86400 / 1e6
            logging.info(f"GPU worker throughput: {speed:.2f} it/s ({speed_mtpd:.2f} mits/d). Processed {total_count} items.")

# This basic structure can work as a template for other similar tasks.
def producer_task(input_queue: mp.Queue, dataset_path: str, output_dir: str, refresh_freq: int):
    """
    This function examples a producer process that reads a dataset and puts (image, question) pairs into an input queue.
    
    Considering recovery of processed records takes different forms in different scenarios, the recovery part is left 
    as a placeholder. Thus we suppose to treat this function as a template to be modified for different datasets and 
    recovery strategies, rather than a ready-to-use function.
    
    Args:
        input_queue: multiprocessing.Queue to which to write (image, question) pairs
        dataset_path: path to the dataset to be processed
        output_dir: directory where output files (e.g., processed records) are stored, used for recovery
        refresh_freq: frequency (in number of processed items) to log speed information
    """
    # ----- load dataset -----
    try:
        
        from core.data.reader import FastLerobotVLReader # for example
        dataset = FastLerobotVLReader(
            root=dataset_path,
            ego_name=None,
            exo_name=None,
            load_all_camera_keys=True,
        )
    except Exception as e:
        logging.error(f"Failed to load dataset from {dataset_path}: {e}")
        input_queue.put(None)
        return

    # -----  try to recover processed record -----
    output_dir = Path(output_dir) 
    # example: if pkl name format is '{dataset_name}_{rank}.pkl'
    # possible_processed_records_paths = list(output_dir.glob(f"{dataset.name}_*.pkl"))
    # ...

    # ----- start processing -----
    last_time = time.time()
    total_frames = len(dataset)
    frame_count = 0
    camera_keys = dataset.camera_keys
    for data in tqdm(dataset, desc=f"[{dataset.name}] Producer"):
        task_desc = data['task_description']

        img = []
        for cam_key in camera_keys:
            img.append(data[cam_key])
        input_queue.put((img, task_desc, 'supp str', None))
        frame_count += 1
        
        if frame_count and frame_count % refresh_freq == 0:
            elapsed = time.time() - last_time
            speed = refresh_freq / elapsed
            rest_frames = total_frames - frame_count
            eta = rest_frames / speed if speed > 1e-5 else -1
            logging.info(f"Producer throughput: {speed:.2f} frames/s. Submitted {frame_count}/{total_frames} frames. ETA: {eta/60:.2f} mins.")
            last_time = time.time()
    # end of input
    input_queue.put(None)
    return

def consumer_task(output_queue: mp.Queue, dataset_path: str, output_dir: str, refresh_freq: int):
    """
    This function defines a consumer process that reads (image, question, generated_text) triples from an output queue
    and saves the results to a file. It also logs speed information periodically.

    Considering recovery of processed records takes different forms in different scenarios, the recovery part is left 
    as a placeholder. Thus we suppose to treat this function as a template to be modified for different datasets and 
    recovery strategies, rather than a ready-to-use function.

    Args:
        output_queue: multiprocessing.Queue from which to read (image, question, generated_text) triples
        dataset_path: path to the dataset being processed, used for naming output files
        output_dir: directory where output files (e.g., processed records) are stored
        refresh_freq: frequency (in number of processed items) to log speed information
    """
    dataset_name = Path(dataset_path).name
    output_dir = Path(output_dir)

    # -----  try to recover processed record -----
    jsl_path = output_dir / f"name.jsonl"
    if jsl_path.exists():
        processed_records = load_jsonlines(jsl_path)
    else:
        processed_records = []
    
    frame_count, success_count = 0, 0
    last_time = time.time()
    while True:
        item = output_queue.get()
        if item is None:
            break
        pil_img, question, generated_text, _supp = item
        # --- try to call a post-process function here ---
        _failed_indices = []
        # generated_text, _failed_indices = postprocess([generated_text], [pil_img]
        # generated_text = generated_text[0]
        processed_records.append({
            "question": question,
            "answer": generated_text
        })
        if len(_failed_indices) == 0: 
            success_count += 1
        elif frame_count < refresh_freq: 
            # save some failed samples
            pass
        frame_count += 1
        if frame_count % refresh_freq == 0:
            write_jsonlines(processed_records, jsl_path)
            speed = refresh_freq / (time.time() - last_time)
            logging.info(
                f"Consumer throughput: {speed:.2f} frames/s. "
                f"Processed {frame_count} frames, {success_count} successful. "
                f"Failed rate: {(frame_count - success_count)/frame_count:.2%}."
            )
            last_time = time.time()
            # save some processed samples
            pass

    write_jsonlines(processed_records, jsl_path)
    logging.info(f"Saved records for {dataset_name} to {jsl_path}. This file has {len(processed_records)} records.")
    return
