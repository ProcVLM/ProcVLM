"""
python tests/test_dataloader.py
"""
import time
import psutil
import torch
import os
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from dotenv import load_dotenv
load_dotenv()

from evqa.data.data_processor import make_supervised_data_module
from evqa.train.argument import DataArguments

# ==============================
# Configs
# ==============================

MODEL_PATH = os.getenv('VLM_PATH_QWEN_TRAIN')
DATASET = "train_s1_8k%100,train_s1_16k%100,train_s2_8k%100,train_s2_16k%100"

BATCH_SIZE = 1
NUM_WORKERS = 16
PREFETCH_FACTOR = 2
PIN_MEMORY = True
PERSISTENT_WORKERS = True

NUM_STEPS = 2000

# ==============================
# Profiler
# ==============================
class DataLoaderProfiler:

    def __init__(self, dataloader):
        self.loader = dataloader
        self.process = psutil.Process(os.getpid())

    # --------------------------------
    # Process tree
    # --------------------------------

    def get_process_tree(self):
        procs = [self.process]
        procs += self.process.children(recursive=True)
        return procs

    # --------------------------------
    # Memory stats
    # --------------------------------

    def get_memory(self):

        rss_total = 0
        pss_total = 0

        procs = self.get_process_tree()

        worker_stats = []

        for p in procs:
            try:
                info = p.memory_info()
                full = p.memory_full_info()

                rss = info.rss
                pss = getattr(full, "pss", rss)

                rss_total += rss
                pss_total += pss

                worker_stats.append(
                    {
                        "pid": p.pid,
                        "rss": rss,
                        "pss": pss,
                    }
                )

            except psutil.NoSuchProcess:
                pass

        return {
            "rss": rss_total / 1024**3,
            "pss": pss_total / 1024**3,
            "workers": worker_stats,
        }

    # --------------------------------
    # DataLoader queue
    # --------------------------------

    def get_queue_size(self, iterator):

        try:
            return iterator._worker_result_queue.qsize()
        except Exception:
            return -1

    # --------------------------------
    # Worker count
    # --------------------------------

    def get_worker_count(self):
        return len(self.process.children())

    # --------------------------------
    # Print stats
    # --------------------------------

    def report(self, step, step_time, iterator):

        mem = self.get_memory()

        queue_size = self.get_queue_size(iterator)
        workers = self.get_worker_count()

        print(
            f"step {step:5d} | "
            f"{step_time:.4f}s | "
            f"workers {workers} | "
            f"queue {queue_size:3d} | "
            f"RSS {mem['rss']:.2f}G | "
            f"PSS {mem['pss']:.2f}G"
        )

# ==============================
# Build processor
# ==============================

processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)

# ==============================
# Build data args (same as train)
# ==============================

data_args = DataArguments(
    dataset_use=DATASET,
    data_packing=True,
    data_flatten=False,

    max_pixels=512*512,
    min_pixels=32*32,

    video_fps=2,
    video_max_frames=512,
    video_min_frames=1,

    video_max_pixels=512*512,
    video_min_pixels=32*32,
)
data_args.model_type = "qwen3vl"

# ==============================
# Create dataset
# ==============================

load_start = time.time()

data_module = make_supervised_data_module(processor, data_args)

dataset = data_module["train_dataset"]
collator = data_module["data_collator"]

print("Dataset size:", len(dataset))

# ==============================
# Create dataloader
# ==============================

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collator,
    num_workers=NUM_WORKERS,
    prefetch_factor=PREFETCH_FACTOR,
    pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT_WORKERS,
)

load_time = time.time() - load_start
print(f"Data loading time: {load_time:.2f}s")

# ==============================
# Benchmark
# ==============================

print("\nStarting dataloader benchmark...\n")

profiler = DataLoaderProfiler(loader)

iterator = iter(loader)

start = time.time()
step_times = []

pss_peak = 0

for step in range(NUM_STEPS):

    t0 = time.time()

    batch = next(iterator)

    # simulate minimal train step
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            _ = batch[k].shape

    step_time = time.time() - t0
    step_times.append(step_time)

    pss_peak = max(pss_peak, profiler.get_memory()["pss"])

    if step % 10 == 0:
        profiler.report(step, step_time, iterator)

total = time.time() - start

print("\n==============================")
print("Benchmark Result")
print("==============================")

print(f"Total time: {total:.2f}s")
print(f"Steps: {NUM_STEPS}")
print(f"Avg step time: {sum(step_times)/len(step_times):.4f}s")
print(f"Throughput: {NUM_STEPS/total:.2f} samples/sec")
print(f"Peak memory (PSS): {pss_peak:.2f} GB")