import time
from evqa.tools.functions import iter_episode_files

if __name__ == "__main__":
    start_time = time.time()
    idx = 0
    for cluster, ds_name, ep in iter_episode_files("/pretrain_data/EVQA/json/Procedural-RAW-0228-coarse", cluster_filter=["a", "b", "c"]):
        idx += 1
        if idx % 10000 == 0:
            dur = time.time() - start_time
            spd = idx / dur
            print(f"Processed {idx} episodes in {dur:.2f} seconds, speed: {spd:.2f} episodes/second")