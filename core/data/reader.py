import json
import re
import time
import logging
import threading
import decord, av, ffmpeg
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple, Union
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from datasets.features.image import Image as HFImage
from fastparquet import ParquetFile
from collections import OrderedDict
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from decord import VideoReader, DECORDError
from core.utils.common import (
    load_json, 
    load_jsonlines, 
    append_jsonlines,
)

# ===== Marcos =====
INFO_PATH = "meta/info.json"
EPISODES_PATH = "meta/episodes.jsonl"
STATS_PATH = "meta/stats.json"
EPISODES_STATS_PATH = "meta/episodes_stats.jsonl"
TASKS_PATH = "meta/tasks.jsonl"
SUB_TASKS_PATH = "meta/sub_tasks.jsonl"
# COTS_PATH = "meta/cots.jsonl"
# BBOXES_PATH = "meta/bboxes.jsonl"


# ===== Default Features =====
logger = logging.getLogger('vl_reader')

class LeRobotDatasetMetadata:
    # modified version of lerobot.dataset.lerobot_dataset.LerobotDatasetMetadata
    # to improve cache efficiency and speed up loading, focusing on local files only
    def __init__(
        self,
        repo_id: str,
        root: Union[str, Path],
    ):
        self.repo_id = repo_id
        self.root = Path(root)

        try:
            self.load_metadata()
        except (FileNotFoundError, NotADirectoryError) as e:
            raise NotImplementedError(f"Cannot load dataset metadata locally: {e})")

    def load_metadata(self):
        self.info = load_info(self.root)
        self.tasks, self.task_to_task_index = load_tasks(self.root)

        # if (self.root/COTS_PATH).exists():
        #     self.cots = load_cots(self.root)
        # else:
        #     self.cots = None

        if (self.root/SUB_TASKS_PATH).exists():
            self.sub_tasks = load_sub_tasks(self.root)
        else:
            self.sub_tasks = None

        # if (self.root/BBOXES_PATH).exists():
        #     self.bboxes = load_bboxes(self.root)
        # else:
        #     self.bboxes = None

        self.episodes = load_episodes(self.root)
        # We removed self.stats and self.episodes_stats because they are either optional or useless

    def get_data_file_path(self, ep_index: int) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.data_path.format(episode_chunk=ep_chunk, episode_index=ep_index)
        return Path(fpath)

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.video_path.format(episode_chunk=ep_chunk, video_key=vid_key, episode_index=ep_index)
        return Path(fpath)

    def get_episode_chunk(self, ep_index: int) -> int:
        return ep_index // self.chunks_size

    @property
    def data_path(self) -> str:
        """Formattable string for the parquet files."""
        return self.info["data_path"]

    @property
    def video_path(self) -> Union[str, None]:
        """Formattable string for the video files."""
        return self.info["video_path"]

    @property
    def robot_type(self) -> Union[str, None]:
        """Robot type used in recording this dataset."""
        return self.info["robot_type"]

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.info["fps"]

    @property
    def features(self) -> dict[str, dict]:
        """All features contained in the dataset."""
        return self.info["features"]

    @property
    def image_keys(self) -> list[str]:
        """Keys to access visual modalities stored as images."""
        return [key for key, ft in self.features.items() if ft["dtype"] == "image"]

    @property
    def video_keys(self) -> list[str]:
        """Keys to access visual modalities stored as videos."""
        return [key for key, ft in self.features.items() if ft["dtype"] == "video"]

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access visual modalities (regardless of their storage method)."""
        return [key for key, ft in self.features.items() if ft["dtype"] in ["video", "image"]]

    @property
    def names(self) -> dict[str, Union[list, dict]]:
        """Names of the various dimensions of vector modalities."""
        return {key: ft["names"] for key, ft in self.features.items()}

    @property
    def shapes(self) -> dict:
        """Shapes for the different features."""
        return {key: tuple(ft["shape"]) for key, ft in self.features.items()}

    @property
    def total_episodes(self) -> int:
        """Total number of episodes available."""
        return self.info["total_episodes"]

    @property
    def total_frames(self) -> int:
        """Total number of frames saved in this dataset."""
        return self.info["total_frames"]

    @property
    def total_tasks(self) -> int:
        """Total number of different tasks performed in this dataset."""
        return self.info["total_tasks"]

    @property
    def total_chunks(self) -> int:
        """Total number of chunks (groups of episodes)."""
        return self.info["total_chunks"]
    
    @property
    def state_dim(self) -> int:
        """Dimension size of observation state."""
        assert len(self.features["observation.state"]["shape"]) == 1

        return self.features["observation.state"]["shape"][0]

    @property
    def action_dim(self) -> int:
        """Dimension size of action."""
        assert len(self.features["action"]["shape"]) == 1
        
        return self.features["action"]["shape"][0]
    
    @property
    def chunks_size(self) -> int:
        """Max number of episodes per chunk."""
        return self.info["chunks_size"]

    def get_task_index(self, task: str) -> Union[int, None]:
        """
        Given a task in natural language, returns its task_index if the task already exists in the dataset,
        otherwise return None.
        """
        return self.task_to_task_index.get(task, None)

    def add_task(self, task: str):
        """
        Given a task in natural language, add it to the dictionary of tasks.
        """
        if task in self.task_to_task_index:
            raise ValueError(f"The task '{task}' already exists and can't be added twice.")

        task_index = self.info["total_tasks"]
        self.task_to_task_index[task] = task_index
        self.tasks[task_index] = task
        self.info["total_tasks"] += 1

        task_dict = {
            "task_index": task_index,
            "task": task,
        }
        append_jsonlines(task_dict, self.root / TASKS_PATH)

    # We removed self.pull_from_repo, self.save_episode, self.update_video_info, classmethod create because they are not needed

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Total episodes: '{self.total_episodes}',\n"
            f"    Total frames: '{self.total_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )

class FastLerobotVLReader:
    """
    A fast and memory-efficient data reader for LeRobot datasets stored in Parquet format.

    Features:
        - Lazy loading of Parquet files with an LRU cache to limit memory usage.
        - On-the-fly decoding of frames from associated MP4 videos.
        - Pre-computation of a global index map for fast item access.
        - Automatic detection of camera keys and task/sub-task descriptions.
    
    Properties:
        - meta: LeRobotDatasetMetadata, metadata of the dataset.
        - loaded_camera_keys: Set[str], set of camera keys being loaded.
        - sub_task_available: bool, whether sub-task annotations are available.
        - exo_camera_available: bool, whether exocentric camera is available.
        - ego_camera_available: bool, whether egocentric camera is available.
        - all_episode_indices: List[int], list of all episode indices in the dataset.
        - all_episode_paths: List[str], list of all episode parquet file paths in the dataset.
        - all_episode_names: List[str], list of all episode parquet file names (Path.stem) in the dataset.

    Functionality:
        - get_episode_range(ep_str): Convert episode string description to (start_index, end_index) tuple.
        - episode_index_to_path(ep_index): Convert episode index to full parquet path.
        - episode_string_to_index(ep_str): Convert episode string description to episode index.
        - set_skip_episode(skip_names): Skip specified episodes by their names (Path.stem).
        - set_skip_episode_by_index(skip_indices): Skip specified episodes by their indices.

    Iterable:
        - __len__(): Returns the total number of frames in the dataset.
        - __getitem__(index): Retrieves a data item by global index or slice. The returned item contains:
            - frame_index: int or np.int64, absolute frame index within the episode
            - task_index: int
            - task: str, task description
            - sub_task_index: int (if available)
            - sub_task: str, sub-task description (if available)
            - task_description: str, either sub-task or task description, prefer sub-task if available
            - exo_image: PIL.Image, image from the exocentric camera (if available)
            - ego_image: PIL.Image, image from the egocentric camera (if available)
            - [other camera keys]: PIL.Image, images from other cameras, named by their camera keys (if load_all_camera_keys is True)
            - episode_path: str, path to the *.parquet file (if return_record_meta is True)
            - episode_len: int, number of frames in the episode (if return_record_meta is True)
            - chunk_name: str, chunk directory name (if return_record_meta is True)
            - episode_name: str, episode file name (if return_record_meta is True)
    """
    def __init__(
        self, 
        root: str, 
        ego_name: Optional[str] = None, 
        exo_name: Optional[str] = None, 
        image_resize: Optional[Tuple[int, int]] = None,
        load_all_camera_keys: bool = False,
        return_record_meta: bool = False,
        cache_num: int = 16, 
        prefetch_num: int = 5,
        use_gpu_decoding: bool = False,
        check_episode_integrity: bool = False,
    ):
        """
        Args:
            root (str): Root directory of the LeRobot dataset.
            ego_name (Optional[str]): Name of the egocentric camera key. If None, auto-detected.
            exo_name (Optional[str]): Name of the exocentric camera key. If None, auto-detected.
            image_resize (Optional[Tuple[int, int]]): If provided, resize all images to this size (width, height). Set width or height to -1 to maintain aspect ratio.
            cache_num (int): Number of Parquet files to keep in memory cache.
            prefetch_num (int): Number of upcoming Parquet files to prefetch in the background.
            load_all_camera_keys (bool): Whether to load all camera keys or just egocentric and exocentric.
            return_record_meta (bool): Whether to return metadata about the record (parquet path, offset, etc.) with each item.
        """
        start_time = time.time()
        self.root = Path(root)
        self.name = self.root.name
        self.meta = LeRobotDatasetMetadata(repo_id=self.name, root=root)

        # --- Reader Settings ---
        self.image_resize = image_resize
        self.load_all_camera_keys = load_all_camera_keys
        self.return_record_meta = return_record_meta
        self.cache_num = cache_num
        self.prefetch_num = prefetch_num
        if self.cache_num <= 0:
            self.cache_num = 1
            logger.warning(f"cache_num should be positive. Setting cache_num to {self.cache_num}.")
        if self.prefetch_num < 0:
            self.prefetch_num = 0
            logger.warning(f"prefetch_num should be non-negative. Setting prefetch_num to {self.prefetch_num}.")
        if self.prefetch_num and not (self.cache_num >= 2 * (self.prefetch_num + 1)):
            self.prefetch_num = (self.cache_num // 2) - 1
            logger.warning(f"prefetch_num+1 should be at most half of cache_num. Setting prefetch_num to {self.prefetch_num}.")
        # if self.load_all_camera_keys and self.prefetch_num != 0:
        #     self.prefetch_num = 0
        #     logger.warning("Using prefetch_num > 0 may reduce performance when load_all_camera_keys is True. Setting prefetch_num to 0.")
        if use_gpu_decoding:
            self.decord_ctx = decord.gpu(0)
        else:
            self.decord_ctx = decord.cpu(0)

        # --- Caches and Threading ---
        self.df_cache = OrderedDict()
        self.frame_cache = OrderedDict()
        self.executor = ThreadPoolExecutor(max_workers=2*(self.prefetch_num+1))
        self.prefetching = set()
        self.load_event = {}

        # --- Prepare Metadata ---
        self.task_index_to_task = self.meta.tasks
        self.sub_task_index_to_sub_task = self.meta.sub_tasks
        if self.sub_task_index_to_sub_task is None:
            logger.warning(f"'sub_tasks.jsonl' not found in {self.name}. Falling back to task descriptions.")
        self._prepare_camera_keys(ego_name, exo_name)
        self._prepare_episode(do_all_checks=check_episode_integrity) # Scan parquets to build index map
        end_time = time.time()
        logger.info(
            f'Initialized Visual Language Reader for dataset "{self.name}". '
            f'Overhead time: {end_time - start_time:.2f}s'
        )

    def __len__(self) -> int:
        return self.total_frames
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self.total_frames)
            return [self._get_by_global_index(i) for i in range(start, stop, step)]
        elif isinstance(index, int):
            return self._get_by_global_index(index)
        raise TypeError(f"Invalid index type: {type(index)}. Must be int or slice.")

    def _get_by_global_index(self, global_index: int) -> Dict[str, Any]:
        if not 0 <= global_index < self.total_frames:
            raise IndexError(f"Index {global_index} is out of bounds for dataset with length {self.total_frames}")
        # Find which parquet file and local index this global index corresponds to
        p_idx, local_index = self._find_parquet_for_index(global_index)
        parquet_info = self.parquet_info[p_idx]
        # Get the DataFrame from cache or load it
        self._maybe_prefetch(p_idx)
        df = self.df_cache[p_idx]
        # Get the specific row of data
        row = df.iloc[local_index]
        item = {}
        # Basic indices
        item['frame_index'] = row['frame_index']
        item['task_index'] = row['task_index']
        if 'sub_task_index' in row:
            item['sub_task_index'] = row['sub_task_index']
        item['task'] = self.task_index_to_task[int(row['task_index'])]
        # Get task description
        if self.sub_task_index_to_sub_task and int(row['sub_task_index']) in self.sub_task_index_to_sub_task:
            item['sub_task'] = self.sub_task_index_to_sub_task[int(row['sub_task_index'])]
        if 'sub_task' in item and item['sub_task']:
            item['task_description'] = item['sub_task']
        else:
            item['task_description'] = item['task']
        # Get egocentric view
        if self.ego_camera_key:
            item['ego_image'] = self._get_image(
                p_idx, local_index,
                row, self.ego_camera_key, parquet_info[self.ego_camera_key]
            )
        # Get exocentric view
        if self.exo_camera_key:
            item['exo_image'] = self._get_image(
                p_idx, local_index,
                row, self.exo_camera_key, parquet_info[self.exo_camera_key]
            )
        # Optionally get all other camera images
        for cam_key in self.loaded_camera_keys:
            if cam_key == self.ego_camera_key:
                item[cam_key] = item['ego_image']
            elif cam_key == self.exo_camera_key:
                item[cam_key] = item['exo_image']
            else:
                item[cam_key] = self._get_image(
                    p_idx, local_index,
                    row, cam_key, parquet_info[cam_key]
                )
        # meta info
        if self.return_record_meta:
            item['episode_path'] = parquet_info['path']
            item['episode_len'] = parquet_info['num_frames']
            item['chunk_name'] = parquet_info['chunk_name']
            item['episode_name'] = parquet_info['episode_name']     
        return item

    def _get_image(self, p_idx: int, local_index: int, row: pd.Series, camera_key: str, video_path: Optional[str]) -> Image.Image:
        """Get image from either embedded bytes or video file, with frame-level caching."""
        frame_cache_key = (p_idx, local_index, camera_key)
        if frame_cache_key in self.frame_cache:
            img = self.frame_cache[frame_cache_key]
        else:
            img = self._get_image_basic(row, camera_key, video_path)
        return img

    def _regularize_image(self, img: Image.Image) -> Image.Image:
        """Convert image to RGB and resize if needed."""
        if (self.image_resize != None):
            aim_w, aim_h = self.image_resize
            if aim_w != -1 or aim_h != -1:
                if aim_w == -1:
                    w, h = img.size
                    aim_w = int(w * (aim_h / h))
                elif aim_h == -1:
                    w, h = img.size
                    aim_h = int(h * (aim_w / w))
                if  (img.size != (aim_w, aim_h)):
                    img = img.resize((aim_w, aim_h), Image.Resampling.LANCZOS)
        return img
    
    def _get_video_length(self, video_path: str) -> int:
        try:
            vr = VideoReader(video_path)
            return len(vr)
        except DECORDError:
            pass  # Fallback to av
        with av.open(video_path) as container:
            return container.streams.video[0].frames

    def _iter_video_frames(self, video_path: str):
        # decord
        try:
            vr = VideoReader(video_path)
            for frame in vr:
                yield frame.asnumpy()
            return
        except DECORDError as e:
            logger.debug(f"Decord failed for {video_path}. Falling back to PyAV. You may ignore this if no more ERROR logs follow. Fault: {e}")
        # PyAV
        try:
            with av.open(video_path, "r") as container:
                video_stream = next(s for s in container.streams if s.type == "video")
                for frame in container.decode(video_stream):
                    yield frame.to_ndarray(format="rgb24")
            return
        except Exception as e:
            logger.warning(f"PyAV failed for {video_path}. Falling back to ffmpeg-python. You may ignore this if no more ERROR logs follow. Fault: {e}")
        # ffmpeg-python fallback
        try:
            probe = ffmpeg.probe(video_path)
            video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
            w, h = int(video_info["width"]), int(video_info["height"])
            out, _ = (
                ffmpeg
                .input(video_path)
                .output("pipe:", format="rawvideo", pix_fmt="rgb24")
                .run(capture_stdout=True, capture_stderr=True)
            )
            buf = memoryview(out)
            frame_size = w * h * 3
            for i in range(0, len(buf), frame_size):
                yield np.frombuffer(buf[i:i+frame_size], np.uint8).reshape(h, w, 3)
            return
        except ffmpeg.Error as e:
            logger.error(f"ffprobe/ffmpeg failed for {video_path}: {e.stderr.decode()}")
            raise

    def _decode_frame(self, video_path: str, frame_index: int) -> np.ndarray:
        # decord
        try:
            vr = VideoReader(video_path)
            return vr[frame_index].asnumpy()
        except (DECORDError, IndexError) as e:
            logger.debug(f"Decord failed to decode frame {frame_index} from {video_path}. Falling back to PyAV. You may ignore this if no more ERROR logs follow. Fault: {e}")
        # PyAV
        try:
            with av.open(video_path) as container:
                stream = container.streams.video[0]
                stream.codec_context.options = {"strict": "experimental"}
                for i, frame in enumerate(container.decode(stream)):
                    if i == frame_index:
                        return frame.to_ndarray(format='rgb24')
        except Exception as e:
            logger.warning(f"PyAV failed to decode frame {frame_index} from {video_path}. Falling back to ffmpeg-python. You may ignore this if no more ERROR logs follow. Fault: {e}")
        # ffmpeg-python fallback
        try:
            out, _ = (
                ffmpeg
                .input(video_path)
                .output("pipe:", format="rawvideo", pix_fmt="rgb24", ss=frame_index / self.meta.fps, vframes=1)
                .run(capture_stdout=True, capture_stderr=True)
            )
            video_info = ffmpeg.probe(video_path)
            video_stream = next(s for s in video_info["streams"] if s["codec_type"] == "video")
            w, h = int(video_stream["width"]), int(video_stream["height"])
            frame = np.frombuffer(out, np.uint8).reshape(h, w, 3)
            return frame
        except ffmpeg.Error as e:
            logger.error(f"ffmpeg failed to decode frame {frame_index} from {video_path}: {e.stderr.decode()}")
            raise

    def _get_image_basic(self, row: pd.Series, camera_key: str, video_path: Optional[str]) -> Image.Image:
        """Decode image from either embedded bytes or video file."""
        if video_path is None:
            img_data = row[camera_key]
            if isinstance(img_data, Image.Image):
                img = img_data
            elif isinstance(img_data, HFImage):
                img = img_data.to_pil()
            elif isinstance(img_data, bytes):
                img = Image.open(BytesIO(img_data))
            elif isinstance(img_data, dict) and 'bytes' in img_data:
                img = Image.open(BytesIO(img_data['bytes']))
            else:
                raise TypeError(f"Unsupported embedded image format: {type(img_data)}")
        else:
            frame = self._decode_frame(video_path, int(row['frame_index']))
            img = Image.fromarray(frame)
        return self._regularize_image(img)

    def _find_parquet_for_index(self, global_index: int) -> Tuple[int, int]:
        """Finds the parquet file index and the local index within that file."""
        # `searchsorted` finds the index where the element should be inserted to maintain order.
        # This is exactly what we need to find which "bin" (parquet file) the index falls into.
        p_idx = np.searchsorted(self.cumulative_frames, global_index, side='right')
        
        # Calculate the local index
        if p_idx == 0:
            local_index = global_index
        else:
            local_index = global_index - self.cumulative_frames[p_idx - 1]
            
        return int(p_idx), int(local_index)

    def _load_df_from_parquet(self, parquet_info: Dict[str, Any]) -> pd.DataFrame:
        """Loads only the necessary columns from a parquet file."""
        required_columns = ['frame_index', 'task_index']
        if self.sub_task_index_to_sub_task is not None:
            required_columns.append('sub_task_index')

        pf = ParquetFile(parquet_info['path'])
        column_names = pf.columns
        needs_rename = False
        for key in self.loaded_camera_keys:
            if key in column_names:
                required_columns.append(key)
            elif key + '.bytes' in column_names:
                required_columns.append(key + '.bytes')
                needs_rename = True
            # use video if neither exists

        logger.debug(f"Loading {parquet_info['path']} with columns: {required_columns}")
        df = pf.to_pandas(columns=required_columns)
        if needs_rename:
            df.rename(columns={col: col[:-6] for col in df.columns if col.endswith('.bytes')}, inplace=True)
        return df
    
    # --- Cache or Prefetch Methods ---
    def _ensure_episode_cache(self, p_idx: int):
        try:
            # --- fast-path: if cached, no-op ---
            if p_idx in self.df_cache:
                return
            # --- load df and frames (slow path) ---
            parquet_info = self.parquet_info[p_idx]
            df = self._load_df_from_parquet(parquet_info)
            frames_to_insert = {}
            for cam_key in self.loaded_camera_keys:
                vpath = parquet_info[cam_key]
                if vpath is not None:
                    for local_idx, frame in zip(range(parquet_info["num_frames"]), self._iter_video_frames(vpath)):
                        frames_to_insert[(p_idx, local_idx, cam_key)] = self._regularize_image(Image.fromarray(frame))
                else:
                    for local_idx in range(parquet_info["num_frames"]):
                        row = df.iloc[local_idx]
                        frames_to_insert[(p_idx, local_idx, cam_key)] = self._get_image_basic(row, cam_key, None)
            # --- commit to cache ---
            self.df_cache[p_idx] = df
            self.frame_cache.update(frames_to_insert)
            # evict old
            if len(self.df_cache) > self.cache_num:
                old_p_idx, old_df = self.df_cache.popitem(last=False)
                for cam_key in self.loaded_camera_keys:
                    for local_idx in range(old_df.shape[0]):
                        self.frame_cache.pop((old_p_idx, local_idx, cam_key), None)
        except Exception as e:
            logger.error(f"Error prefetching parquet index {p_idx}: {e}")
        finally:
            self.prefetching.discard(p_idx)
            self.load_event[p_idx].set()
        
    def _maybe_prefetch(self, p_idx: int):
        """ toggle prefetching for next few parquets """
        # submit current task if not yet submitted
        if p_idx not in self.prefetching:
            self.prefetching.add(p_idx)
            self.load_event.setdefault(p_idx, threading.Event())
            self.executor.submit(self._ensure_episode_cache, p_idx)
        # submit future prefetched tasks
        for offset in range(1, self.prefetch_num + 1):
            target_idx = p_idx + offset
            if target_idx >= len(self.parquet_info):
                break
            if target_idx not in self.prefetching:
                self.prefetching.add(target_idx)
                self.load_event.setdefault(target_idx, threading.Event())
                self.executor.submit(self._ensure_episode_cache, target_idx)
        # blocking behavior
        self.load_event[p_idx].wait()
            
    # --- Metadata Preparation Methods (largely from your original code, with fixes) ---
    def _prepare_camera_keys(self, ego_name=None, exo_name=None):
        self.camera_keys = self.meta.camera_keys
        assert len(self.camera_keys) > 0, f"No camera keys found in {self.name}."

        self.loaded_camera_keys = set()
        if self.load_all_camera_keys:
            self.loaded_camera_keys = set(self.camera_keys)

        self.ego_camera_key = ego_name if ego_name in self.camera_keys else None
        if self.ego_camera_key is None:
            for key in self.camera_keys:
                if any(k in key.lower() for k in ['wrist', 'ego', 'interior']):
                    self.ego_camera_key = key
                    self.loaded_camera_keys.add(self.ego_camera_key)
                    break
        else:
            self.loaded_camera_keys.add(self.ego_camera_key)
        
        self.exo_camera_key = exo_name if exo_name in self.camera_keys else None
        if self.exo_camera_key is None:
            for key in self.camera_keys:
                if any(k in key.lower() for k in ['front', 'exo', 'exterior', 'global']):
                    self.exo_camera_key = key
                    break
            if self.exo_camera_key is None:
                self.exo_camera_key = self.camera_keys[0] # A reasonable fallback
                logger.warning(f"Could not auto-detect exterior camera. Defaulting to '{self.exo_camera_key}'.")
        self.loaded_camera_keys.add(self.exo_camera_key)

        logger.info(f"Using egocentric camera: '{self.ego_camera_key}', exterior camera: '{self.exo_camera_key}'")

    def _prepare_episode(self, do_all_checks: bool = False):
        # parquet_paths = load_parquet(self.root)
        # self.check_parquet_integrity(parquet_paths)
        self.parquet_info = []
        cumulative_frames = []
        total_frames = 0
        
        if do_all_checks:
            iterable = tqdm(self.all_episode_paths, desc=f"Checking parquets in {self.name}")
        else:
            iterable = self.all_episode_paths
        for p_str in iterable:
            episode_name = Path(p_str).stem
            chunk_name = Path(p_str).parent.stem
            pf = ParquetFile(p_str)
            num_frames = pf.count()
            total_frames += num_frames
            cumulative_frames.append(total_frames)
            
            column_names = pf.columns
            for i in range(len(column_names)):
                if column_names[i].endswith('.bytes'):
                    column_names[i] = column_names[i][:-6]
            info = {'path': p_str, 'num_frames': num_frames, 'columns': column_names, 'chunk_name': chunk_name, 'episode_name': episode_name}
            
            def check_video(camera_key):
                    if not camera_key or camera_key in column_names:
                        return None
                    video_path = self.root / "videos" / chunk_name / camera_key / f"{episode_name}.mp4"
                    assert video_path.exists(), f"Parquet {p_str} requires video {video_path}, but it was not found."
                    if do_all_checks:
                        video_len = self._get_video_length(str(video_path))
                        if video_len < num_frames:
                            raise RuntimeError(f"Video {video_path} has fewer frames ({video_len}) than parquet {p_str} ({num_frames}).")
                        elif video_len > num_frames:
                            logger.warning(f"Video {video_path} has more frames ({video_len}) than parquet {p_str} ({num_frames}). Using video.")
                    return str(video_path)

            if self.load_all_camera_keys:
                for cam_key in self.camera_keys:
                    info[cam_key] = check_video(cam_key)
            else:
                info[self.ego_camera_key] =  check_video(self.ego_camera_key)
                info[self.exo_camera_key] = check_video(self.exo_camera_key)

            self.parquet_info.append(info)

        self.total_frames = total_frames
        self.cumulative_frames = np.array(cumulative_frames, dtype=np.int64)
        logger.info(f"Found {len(self.all_episode_paths)} parquet files in {self.name}, total frames: {self.total_frames}")
    
    # --- Functions ---
    @property
    def sub_task_available(self) -> bool:
        return self.sub_task_index_to_sub_task is not None
    
    @property
    def exo_camera_available(self) -> bool:
        return self.exo_camera_key is not None

    @property
    def ego_camera_available(self) -> bool:
        return self.ego_camera_key is not None

    @property
    def all_episode_indices(self) -> List[int]:
        return sorted(self.meta.episodes.keys())

    def episode_index_to_path(self, ep_index: int) -> Path:
        """
        Convert episode index to full parquet path.
        """
        chunk_index = self.meta.get_episode_chunk(ep_index)
        parquet_path = self.root / self.meta.data_path.format(episode_chunk=chunk_index, episode_index=ep_index)
        return parquet_path

    def episode_string_to_index(self, ep_str: str) -> int:
        """ 
        Convert episode string description to episode index. 
        This string can either be a full path or just the episode file name. 
        """
        ep_path = Path(ep_str)
        if ep_path.suffix == '.parquet':
            ep_name = ep_path.stem
        else:
            ep_name = ep_str
        match = re.search(r'episode_(\d+)$', ep_name)
        if not match:
            raise ValueError(f"Invalid episode string description format: {ep_str}")
        return int(match.group(1))
    
    def get_episode_range(self, ep_str: str) -> Tuple[int, int]:
        """ 
        Convert episode string description to (start_index, end_index) tuple. 
        This string can either be a full path or just the episode file name. 
        """
        ep_name = Path(ep_str).stem if Path(ep_str).suffix == '.parquet' else ep_str
        for p_idx, info in enumerate(self.parquet_info):
            if info["episode_name"] == ep_name:
                start_index = self.cumulative_frames[p_idx - 1] if p_idx > 0 else 0
                end_index = self.cumulative_frames[p_idx]
                return int(start_index), int(end_index)
        raise ValueError(f"Episode '{ep_str}' not found in the dataset.")

    def get_episode_name_by_global_index(self, global_index: int) -> str:
        """ 
        Given a global frame index, return the episode name (Path.stem) it belongs to.
        """
        p_idx, _ = self._find_parquet_for_index(global_index)
        return self.parquet_info[p_idx]["episode_name"]

    @property
    def all_episode_paths(self) -> List[str]:
        parquet_paths = []
        for ep_index in self.all_episode_indices:
            parquet_path = self.episode_index_to_path(ep_index)
            if not parquet_path.exists():
                raise FileNotFoundError(f"Expected parquet file {parquet_path} does not exist.")
            parquet_paths.append(str(parquet_path))
        return parquet_paths
    
    @property
    def all_episode_names(self) -> List[str]:
        return [Path(p).stem for p in self.all_episode_paths]

    def set_skip_episode(self, skip_names: List[str]):
        """
        Skip the specified episodes, given by their names (Path.stem), remove their parquet info
        """
        # clean parquet_info
        new_parquet_info = []
        skipped_names = []
        for info in self.parquet_info:
            if info["episode_name"] in skip_names:
                skipped_names.append(info["episode_name"])
            else:
                new_parquet_info.append(info)
        self.parquet_info = new_parquet_info
        # recompute cumulative_frames and total_frames
        cumulative_frames = []
        total_frames = 0
        for info in self.parquet_info:
            total_frames += info["num_frames"]
            cumulative_frames.append(total_frames)
        self.total_frames = total_frames
        self.cumulative_frames = np.array(cumulative_frames, dtype=np.int64)
        logger.info(
            f"Skipped {len(skipped_names)} episodes. "
            f"Remaining {len(self.parquet_info)} parquet files, total frames: {self.total_frames}."
        )
    
    def set_skip_episode_by_index(self, skip_indices: List[int]):
        """
        Skip the specified episodes, given by their index, remove their parquet info
        """
        ep_paths = [self.episode_index_to_path(idx) for idx in skip_indices]
        skip_names = {p.stem for p in ep_paths}
        self.set_skip_episode(skip_names)


# ===== Utility Functions ===    
def clean_string(s: str) -> str:
    return re.sub(r"[\x00-\x1F\x7F]", "", s)

# copied from lerobot.datasets.utils
def load_info(local_dir: Path) -> dict:
    info = load_json(local_dir / INFO_PATH)
    for ft in info["features"].values():
        ft["shape"] = tuple(ft["shape"])
    return info

# We disabled sorting to make it faster
def load_tasks(local_dir: Path) -> tuple[dict, dict]:
    tasks = load_jsonlines(local_dir / TASKS_PATH)
    for item in tasks: item["task"] = clean_string(item["task"])
    # tasks = {item["task_index"]: item["task"] for item in sorted(tasks, key=lambda x: x["task_index"])}
    tasks = {item["task_index"]: item["task"] for item in tasks}
    task_to_task_index = {task: task_index for task_index, task in tasks.items()}
    return tasks, task_to_task_index

# def load_cots(local_dir: Path) -> dict:
#     cots = load_jsonlines(local_dir / COTS_PATH)
#     # cots = {item["cot_index"]: item["cot"] for item in sorted(cots, key=lambda x: x["cot_index"])}
#     cots = {item["cot_index"]: item["cot"] for item in cots}
#     return cots

def load_sub_tasks(local_dir: Path) -> dict:
    sub_tasks = load_jsonlines(local_dir / SUB_TASKS_PATH)
    # sub_tasks = {item["sub_task_index"]: item["sub_task"] for item in sorted(sub_tasks, key=lambda x: x["sub_task_index"])}
    sub_tasks = {item["sub_task_index"]: item["sub_task"] for item in sub_tasks}
    return sub_tasks

# def format_bboxes(bboxes: list) -> str:
#     if len(bboxes) == 0:
#         return ""
#     for bbox_dict in bboxes:
#         bbox_dict["bbox_2d"] = [round(number, 3) for number in bbox_dict["bbox_2d"]]
#     bboxes_str = json.dumps(bboxes, indent=2, ensure_ascii=False)
#     return bboxes_str

# def load_bboxes(local_dir: Path) -> dict:
#     bboxes = load_jsonlines(local_dir / BBOXES_PATH)
#     # bboxes = {item["bbox_index"]: format_bboxes(item["bbox"]) for item in sorted(bboxes, key=lambda x: x["bbox_index"])}
#     bboxes = {item["bbox_index"]: item["bbox"] for item in bboxes}
#     return bboxes

def load_episodes(local_dir: Path) -> dict:
    episodes = load_jsonlines(local_dir / EPISODES_PATH)
    # return {item["episode_index"]: item for item in sorted(episodes, key=lambda x: x["episode_index"])}
    return {item["episode_index"]: item for item in episodes}

