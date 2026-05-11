import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
from enum import IntFlag
from dotenv import load_dotenv
from core.utils.common import load_jsonlines
from core.data.reader import FastLerobotVLReader
from core.data.generals import EGO_CAMERA_MAP, EXO_CAMERA_MAP

load_dotenv()
ANNOTATION_ROOT = os.getenv('ANNOTATION_ROOT')

class AnnotationLoader:
    sub_task_dir_temp = (Path(ANNOTATION_ROOT) / '{version}/sub_task/').as_posix()
    cot_dir_temp = (Path(ANNOTATION_ROOT) / '{version}/cot/').as_posix()
    bbox_dir_temp = (Path(ANNOTATION_ROOT) / '{version}/bbox/').as_posix()

    sub_task_cache_episode_name: str  = ''
    sub_task_cache: Dict[int, Optional[Dict[str, Any]]] = {}

    cot_cache_episode_name: str  = ''
    cot_cache: Dict[int, Optional[Dict[str, Any]]] = {}

    bbox_cache_episode_name: str  = ''
    bbox_cache: Dict[Tuple[int, str], Optional[Dict[str, Any]]] = {}

    planned_action_cache_episode_name: str  = ''
    planned_action_cache: Dict[int, Optional[Dict[str, Any]]] = {}

    def __init__(self, subtask_dir: Optional[str] = None, cot_dir: Optional[str] = None, bbox_dir: Optional[str] = None, versions: Optional[List[str]] = ['v3', 'v2', 'v1']):
        self.subtask_dir = subtask_dir
        self.cot_dir = cot_dir
        self.bbox_dir = bbox_dir
        self.versions = versions

    def set_dataset(self, dataset: Path | str):
        self.dataset_name = Path(dataset).name if isinstance(dataset, str) else dataset.name

    @property
    def dataset_sub_task_dir(self) -> Optional[Path]:
        if self.subtask_dir and self.dataset_name:
            return Path(self.subtask_dir) / self.dataset_name
        for version in self.versions:
            subtask_dir = self.sub_task_dir_temp.format(version=version)
            candidate_path = Path(subtask_dir) / self.dataset_name
            if candidate_path.exists():
                return candidate_path
        return None

    @property
    def dataset_cot_dir(self) -> Optional[Path]:
        if self.cot_dir and self.dataset_name:
            return Path(self.cot_dir) / self.dataset_name
        for version in self.versions:
            cot_dir = self.cot_dir_temp.format(version=version)
            candidate_path = Path(cot_dir) / self.dataset_name
            if candidate_path.exists():
                return candidate_path
        return None

    @property
    def dataset_bbox_dir(self) -> Optional[Path]:
        if self.bbox_dir and self.dataset_name:
            return Path(self.bbox_dir) / self.dataset_name
        for version in self.versions:
            bbox_dir = self.bbox_dir_temp.format(version=version)
            candidate_path = Path(bbox_dir) / self.dataset_name
            if candidate_path.exists():
                return candidate_path
        return None

    def get_sub_task(self, episode_name: str, frame_id: int) -> Optional[Any]:
        if not self.dataset_sub_task_dir:
            return None
        if self.sub_task_cache_episode_name != episode_name:
            self.sub_task_cache = {}
            file_path = self.dataset_sub_task_dir / f'{episode_name}.sub_task.jsonl'
            if file_path.exists():
                jl = load_jsonlines(file_path)
                self.sub_task_cache = {item['frame_id']: item['sub_task'] for item in jl}
            self.sub_task_cache_episode_name = episode_name
        return self.sub_task_cache.get(frame_id, None)

    def get_cot(self, episode_name: str, frame_id: int) -> Optional[Any]:
        if not self.dataset_cot_dir:
            return None
        if self.cot_cache_episode_name != episode_name:
            self.cot_cache = {}
            file_path = self.dataset_cot_dir / f'{episode_name}.cot.jsonl'
            if file_path.exists():
                jl = load_jsonlines(file_path)
                self.cot_cache = {item['frame_id']: item['cot'] for item in jl}
            self.cot_cache_episode_name = episode_name
        return self.cot_cache.get(frame_id, None)

    def get_bbox_item(self, episode_name: str, frame_id: int, camera_key: str) -> Optional[Any]:
        if not self.dataset_bbox_dir:
            return None
        if self.bbox_cache_episode_name != episode_name:
            self.bbox_cache = {}
            file_path = self.dataset_bbox_dir / f'{episode_name}.bbox.jsonl'
            if file_path.exists():
                jl = load_jsonlines(file_path)
                self.bbox_cache = {(item['frame_id'], item['camera_key']): item for item in jl}
            self.bbox_cache_episode_name = episode_name
        return self.bbox_cache.get((frame_id, camera_key), None)

    def get_bbox(self, episode_name: str, frame_id: int, camera_key: str) -> Optional[Any]:
        bbox_item = self.get_bbox_item(episode_name, frame_id, camera_key)
        return bbox_item['bboxes'] if bbox_item else None

    def get_planned_actions(self, episode_name: str, frame_id: int) -> Optional[Any]:
        if not self.dataset_sub_task_dir:
            return None
        if self.planned_action_cache_episode_name != episode_name:
            self.planned_action_cache = {}
            file_path = self.dataset_sub_task_dir / f'{episode_name}.sub_task.jsonl'
            if file_path.exists():
                jl = load_jsonlines(file_path)
                # sort by frame_id to ensure order
                jl.sort(key=lambda x: x['frame_id'])
                # planned actions = unique list of existing 'sub_task' after current frame_id, excluding 'done', 'not done' or '', within the same task
                current_task = None
                planned_actions = []
                for item in reversed(jl):
                    if item['task'] != current_task:
                        current_task = item['task']
                        planned_actions = []
                    if item['sub_task'] not in ['done', 'not done', ''] and item['sub_task'] not in planned_actions:
                        planned_actions.insert(0, item['sub_task']) # Prepend to maintain order
                    self.planned_action_cache[item['frame_id']] = planned_actions.copy()
            self.planned_action_cache_episode_name = episode_name
        return self.planned_action_cache.get(frame_id, None)

    def get_ecot(self, episode_name: str, frame_id: int, camera_key: str) -> Optional[Dict[str, Any]]:
        sub_task = self.get_sub_task(episode_name, frame_id)
        if sub_task == 'done':
            return {
                'cot': cot,
                'finished': True
            }
        cot = self.get_cot(episode_name, frame_id)
        planned_actions = self.get_planned_actions(episode_name, frame_id)
        bbox = self.get_bbox(episode_name, frame_id, camera_key)
        if sub_task is None or cot is None or bbox is None or len(planned_actions) == 0:
            return None
        return {
            'cot': cot,
            'finished': False,
            'planned actions': planned_actions,
            'target objects': bbox
        }

    def get_ecot_wo_reasoning(self, episode_name: str, frame_id: int, camera_key: str) -> Optional[Dict[str, Any]]:
        sub_task = self.get_sub_task(episode_name, frame_id)
        if sub_task == 'done':
            return {
                'finished': True
            }
        planned_actions = self.get_planned_actions(episode_name, frame_id)
        bbox = self.get_bbox(episode_name, frame_id, camera_key)
        if sub_task is None or bbox is None or len(planned_actions) == 0:
            return None
        return {
            'finished': False,
            'planned actions': planned_actions,
            'target objects': bbox
        }

class AnnotatedLerobotVLReader:
    class RunMode(IntFlag):
        ECoT               = 0        # 0 as unusual case
        ECoT_wo_reasoning  = 1 << 0   # 000001
        SUB_TASK           = 1 << 1   # 000010
        COT                = 1 << 2   # 000100
        BBOX               = 1 << 3   # 001000
        PLANNED_ACTIONS    = 1 << 4   # 010000

    def __init__(self, root: str, load_all_camera_keys: bool=False, **kwargs):
        if not load_all_camera_keys:
            self.vl_reader = FastLerobotVLReader(
                root=root,
                ego_name=EGO_CAMERA_MAP.get(Path(root).name, None),
                exo_name=EXO_CAMERA_MAP.get(Path(root).name, None),
                return_record_meta=True
            )
        else:
            self.vl_reader = FastLerobotVLReader(
                root=root,
                load_all_camera_keys=True,
                return_record_meta=True
            )
        self.annotation_loader = AnnotationLoader(**kwargs)
        self.annotation_loader.set_dataset(root)

    def set_run_mode(self, mode: Union[int, List[str]]):
        if isinstance(mode, int):
            self.run_mode = mode
        elif isinstance(mode, list):
            mode_flag = 0
            for m in mode:
                if m == 'ECoT':
                    mode_flag |= self.RunMode.ECoT
                elif m == 'ECoT_wo_reasoning':
                    mode_flag |= self.RunMode.ECoT_wo_reasoning
                elif m == 'SUB_TASK':
                    mode_flag |= self.RunMode.SUB_TASK
                elif m == 'COT':
                    mode_flag |= self.RunMode.COT
                elif m == 'BBOX':
                    mode_flag |= self.RunMode.BBOX
                elif m == 'PLANNED_ACTIONS':
                    mode_flag |= self.RunMode.PLANNED_ACTIONS
            self.run_mode = mode_flag
        else:
            raise ValueError("mode must be int or List[str]")

    def __len__(self) -> int:
        return self.vl_reader.total_frames

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self.__len__())
            return [self._get_by_index(i) for i in range(start, stop, step)]
        elif isinstance(index, int):
            return self._get_by_index(index)
        raise TypeError(f"Invalid index type: {type(index)}. Must be int or slice.")

    def _get_by_index(self, index: int) -> Dict[str, Any]:
        data = self.vl_reader[index]
        data['exo_camera_key'] = self.vl_reader.exo_camera_key
        if self.vl_reader.ego_camera_key:
            data['ego_camera_key'] = self.vl_reader.ego_camera_key
        data['loaded_camera_keys'] = self.vl_reader.loaded_camera_keys
        episode_name = data['episode_name']
        frame_id = int(data['frame_index'])
        if self.run_mode & self.RunMode.ECoT:
            camera_key = self.vl_reader.exo_camera_key
            _ecot = self.annotation_loader.get_ecot(episode_name, frame_id, camera_key)
            ecot = {
                'task': data['task'],
                'cot': _ecot['cot'],
                'finished': _ecot['finished'],
                'planned actions': _ecot['planned actions'],
                'target objects': _ecot['target objects'],
            }
            data['ecot'] = ecot
        if self.run_mode & self.RunMode.ECoT_wo_reasoning:
            camera_key = self.vl_reader.exo_camera_key
            _ecot_wo_reasoning = self.annotation_loader.get_ecot_wo_reasoning(episode_name, frame_id, camera_key)
            ecot_wo_reasoning = {
                'task': data['task'],
                'finished': _ecot_wo_reasoning['finished'],
                'planned actions': _ecot_wo_reasoning['planned actions'],
                'target objects': _ecot_wo_reasoning['target objects'],
            }
            data['ecot_wo_reasoning'] = ecot_wo_reasoning
        if self.run_mode & self.RunMode.SUB_TASK:
            sub_task = self.annotation_loader.get_sub_task(episode_name, frame_id)
            data['sub_task'] = sub_task
        if self.run_mode & self.RunMode.COT:
            cot = self.annotation_loader.get_cot(episode_name, frame_id)
            data['cot'] = cot
        if self.run_mode & self.RunMode.BBOX:
            camera_keys = self.vl_reader.loaded_camera_keys
            data['bbox'] = {}
            for camera_key in camera_keys:
                bbox = self.annotation_loader.get_bbox(episode_name, frame_id, camera_key)
                data['bbox'][camera_key] = bbox
        if self.run_mode & self.RunMode.PLANNED_ACTIONS:
            planned_actions = self.annotation_loader.get_planned_actions(episode_name, frame_id)
            data['planned_actions'] = planned_actions
        return data

