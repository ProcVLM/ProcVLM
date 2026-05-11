import os
from pathlib import Path
from typing import List, Tuple

def iter_episode_files(root: Path, cluster_filter: List[str]=None):
    """
    root/cluster/ds_name.jsonl 
        OR 
    root/cluster/ds_name/episode_xxxxxx.jsonl
    """
    root = str(root)
    iterable = os.listdir(root)
    if cluster_filter is not None:
        iterable = [c for c in iterable if c in cluster_filter]
    for cluster in iterable:
        cluster_dir = os.path.join(root, cluster)
        if not os.path.isdir(cluster_dir):
            continue
        with os.scandir(cluster_dir) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(".jsonl"):
                    yield cluster, entry.name[:-6], Path(entry.path)
                elif entry.is_dir():
                    ds_name = entry.name
                    ds_dir = os.path.join(cluster_dir, ds_name)
                    with os.scandir(ds_dir) as it2:
                        for entry2 in it2:
                            if entry2.is_file() and entry2.name.endswith(".jsonl"):
                                yield cluster, ds_name, Path(entry2.path)