import random
import logging
import argparse
import gradio as gr
from PIL.Image import Image
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
from core.utils.common import append_jsonlines, load_jsonlines
from evqa.tools.data_picker import AnnotatedLerobotVLReader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# global variables
templates_dir: str
data_output_dir: str
json_output_path: str
selection_path: Optional[str] = None
_cached_templates: Dict[str, List[str]] = {}


# --- 1. Data Generator  ---
def episode_data_generator(datasets: List[str], task_ids: List[str]):
    """ Data generator for episodes from multiple datasets and task IDs. """
    global selection_path
    if selection_path:
        selection_jl = load_jsonlines(selection_path)
        selection = {}
        for item in selection_jl:
            dn = item['dataset_name']
            en = item['episode_name']
            if dn not in selection:
                selection[dn] = set()
            selection[dn].add(en)
    else:
        selection = None
    
    for dataset_path in datasets:
        dataset_name = Path(dataset_path).name
        try:
            reader = AnnotatedLerobotVLReader(root=dataset_path, versions=['v2'])
            reader.set_run_mode(['SUB_TASK', 'PLANNED_ACTIONS'])
        except Exception as e:
            logging.error(f"Failed to load dataset {dataset_name}: {e}")
            continue
        
        # 遍历所有 Episode
        for episode_name in reader.vl_reader.all_episode_names:
            if selection is not None and (dataset_name not in selection or episode_name not in selection[dataset_name]):
                continue

            try:
                s, t = reader.vl_reader.get_episode_range(episode_name)
                episode_frames = reader[s:t]
                episode_stats = {
                    'len': len(episode_frames),
                    'fps': reader.vl_reader.meta.fps,
                }
                
                # --- 原有的数据预处理逻辑 Start ---
                start_fid = 0
                task_splits = []
                for fid in range(1, episode_stats['len']):
                    if episode_frames[fid]['task'] != episode_frames[fid - 1]['task']:
                        task_splits.append((start_fid, fid, episode_frames[fid - 1]['task']))
                        start_fid = fid
                task_splits.append((start_fid, episode_stats['len'], episode_frames[-1]['task']))
                if len(task_splits) > 1:
                    continue
                episode_stats['task_splits'] = task_splits
                
                episode_stats['sub_task_splits'] = {}
                episode_stats['plan'] = {}
                for start_fid, end_fid, task in task_splits:
                    episode_splits = []
                    episode_plan = []
                    current_sub_task_start_fid = start_fid
                    for fid in range(start_fid + 1, end_fid):
                        if episode_frames[fid]['sub_task'] != episode_frames[fid - 1]['sub_task']:
                            episode_splits.append((current_sub_task_start_fid, fid, episode_frames[fid - 1]['sub_task']))
                            episode_plan.append(episode_frames[fid - 1]['sub_task'])
                            current_sub_task_start_fid = fid
                    episode_splits.append((current_sub_task_start_fid, end_fid, episode_frames[end_fid - 1]['sub_task']))
                    episode_plan.append(episode_frames[end_fid - 1]['sub_task'])
                    episode_stats['sub_task_splits'][task] = episode_splits
                    episode_stats['plan'][task] = episode_plan
                # --- 原有的数据预处理逻辑 End ---

                # 对每个 task_id 生成一次数据 (为了保持原有逻辑)
                for task_id in task_ids:
                    yield {
                        "dataset_name": dataset_name,
                        "episode_name": episode_name,
                        "episode_frames": episode_frames,
                        "episode_stats": episode_stats,
                        "task_id": task_id
                    }
                    
            except Exception as e:
                logging.error(f"Error processing episode {episode_name}: {e}")
                continue


# --- 2. 辅助函数 ---
def load_templates(template_name: str) -> List[str]:
    global _cached_templates
    if template_name in _cached_templates:
        return _cached_templates[template_name]
    template_path = Path(templates_dir) / f"{template_name}.txt"
    if not template_path.exists():
        raise FileNotFoundError(f"Template file '{template_path}' not found.")
    with open(template_path, 'r') as f:
        templates = [line.strip() for line in f if line.strip()]
    _cached_templates[template_name] = templates
    return templates

def append_QA_to_json(record: Dict[str, Any]):
    global json_output_path
    append_jsonlines(record, json_output_path)

def ensure_image_on_disk(dataset_name: str, camera_key: str, episode_name: str, frame_id: int, image_data: Image) -> str:
    """make sure the image is saved on disk, return the relative path to data_output_dir"""
    possible_path = Path(data_output_dir) / "images" / dataset_name / camera_key / episode_name / f"{frame_id:06d}.jpg"
    if possible_path.exists():
        return str(possible_path.relative_to(data_output_dir))
    save_path = possible_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image_data.save(save_path)
    return str(save_path.relative_to(data_output_dir))

def ensure_video_on_disk(dataset_name: str, camera_key: str, episode_name: str, start_fid: int, end_fid: int, frame_images: List[Image], fps: int) -> str:
    """make sure the video is saved on disk, return the relative path to data_output_dir"""
    possible_path = Path(data_output_dir) / "videos" / dataset_name / camera_key / episode_name / f"{start_fid:06d}_{end_fid:06d}.mp4"
    if possible_path.exists():
        return str(possible_path.relative_to(data_output_dir))
    save_path = possible_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    images_to_video(frame_images, save_path, fps=fps, show_log=False)
    return str(save_path.relative_to(data_output_dir))


# --- 3. UI 类 (封装状态和逻辑) ---
class InspectorUI:
    def __init__(self, data_gen):
        self.data_gen = data_gen
        
        # 当前 Episode 的状态缓存
        self.current_data = None
        self.estimated_frame_progress = []
        self.templates = []
        self.total_frames = 0
        self.fps = 30
        
    def load_next_episode(self):
        """从生成器获取下一个 Episode 的数据并更新内部状态"""
        try:
            # 获取下一个数据包
            data = next(self.data_gen)
            self.current_data = data
            
            # 解包数据
            episode_stats = data['episode_stats']
            episode_frames = data['episode_frames']
            task_id = data['task_id']
            
            self.total_frames = len(episode_frames)
            self.fps = episode_stats['fps']
            self.templates = load_templates(f'procedural_{task_id}')

            # --- 进度计算逻辑 (Pre-calculation) ---
            self.estimated_frame_progress = []
            self.all_subtasks = []
            for start_fid, end_fid, task in episode_stats['task_splits']:
                self.all_subtasks.extend(episode_stats['plan'][task])
                sub_tasks = episode_stats['sub_task_splits'].get(task, [])
                len_sub_tasks = len(sub_tasks)
                
                # Filter logic
                valid_count = 0
                for _, _, sub_task in sub_tasks:
                    if sub_task and sub_task != 'done':
                        valid_count += 1
                
                # Simple progress mapping
                sub_task_offsets = {}
                for idx, (st, ed, sub_task) in enumerate(sub_tasks):
                    sub_task_offsets[sub_task] = (idx / max(1, valid_count), st, ed)

                for fid in range(start_fid, end_fid):
                    # 简化的进度计算，防止过于复杂导致 UI 卡顿
                    curr_sub = episode_frames[fid]['sub_task']
                    if curr_sub in sub_task_offsets:
                        offset, st_start, st_end = sub_task_offsets[curr_sub]
                        rel = (fid - st_start) / max(1, st_end - st_start)
                        prog = offset + (rel / max(1, valid_count))
                        self.estimated_frame_progress.append(min(int(prog * 100), 100))
                    else:
                        self.estimated_frame_progress.append(0)

            # 补齐长度
            while len(self.estimated_frame_progress) < self.total_frames:
                self.estimated_frame_progress.append(100)

            # 返回 UI 更新对象
            title_str = f"Dataset: {data['dataset_name']} | Episode: {data['episode_name']} | TaskID: {task_id}"
            
            # Reset UI components
            return (
                title_str,                  # Title Markdown
                gr.update(value=0, maximum=self.total_frames-1), # Slider
                "Loaded new episode.",      # Status
                gr.update(interactive=True), # Enable Save
                gr.update(visible=True)     # Main Column visible
            )
            
        except StopIteration:
            return (
                "## All Episodes Finished! 🎉", 
                gr.update(maximum=0), 
                "Done", 
                gr.update(interactive=False),
                gr.update(visible=False)
            )
        except Exception as e:
            logging.error(f"Error loading next: {e}")
            return f"Error: {e}", gr.update(), "Error", gr.update(), gr.update()

    def get_frame_info(self, idx):
        if not self.current_data:
            return None, "No Data", "", "", ""
            
        idx = int(idx)
        # Boundary check
        idx = max(0, min(idx, self.total_frames - 1))
        
        frame_data = self.current_data['episode_frames'][idx]
        
        # Plans
        plans = frame_data.get('planned_actions', [])
        plan_str = "\n".join([f"- {p}" for p in plans]) if plans else "None"
        
        # Info text
        info = (
            f"**Task:** {frame_data['task']}\n\n"
            f"**Sub-task:** {frame_data['sub_task']}\n\n"
            f"**Progress:** {self.estimated_frame_progress[idx]}%\n\n"
            f"**Frame:** {idx} / {self.total_frames - 1}\n\n"
            f"**All Sub-tasks:**\n" + "\n".join([f"- {st}" for st in self.all_subtasks])
        )
        q_input = self.get_random_question()
        q_input = q_input.format(task=frame_data['task'])
        return (
            frame_data['exo_image'], 
            info, 
            plan_str, 
            q_input, 
            f"{self.estimated_frame_progress[idx]}%"
        )

    def get_random_question(self):
        if self.templates:
            return random.choice(self.templates)
        return "Describe the action."

    def calculate_data_range(self, end_idx, data_type):
        configs = {
            "Single Image": (1, False),
            "Image Sequence (4 frames)": (4, False),
            "Image Sequence (8 frames)": (8, False),
            "Video (2s)": (int(2 * self.fps), True),
            "Video (4s)": (int(4 * self.fps), True),
            "Video (Full Task)": (-1, True)
        }
        duration, is_video = configs.get(data_type, (1, False))
        current_fid = int(end_idx)
        
        if duration == -1:
            curr_sub_task = self.current_data['episode_frames'][current_fid]['sub_task']
            start_fid = current_fid
            while start_fid > 0 and self.current_data['episode_frames'][start_fid-1]['sub_task'] == curr_sub_task:
                start_fid -= 1
        else:
            start_fid = max(0, current_fid - duration + 1)
        return start_fid, current_fid, is_video

    def update_preview_label(self, idx, data_type):
        if not self.current_data: return ""
        s, e, is_vid = self.calculate_data_range(idx, data_type)
        type_str = "Video" if is_vid else "Image(s)"
        return f"Preview: Will save {type_str} from Frame {s} to {e} (Count: {e - s + 1})"

    def save_annotation(self, idx, question, answer, data_type):
        if not self.current_data: return "No episode loaded.", gr.update()
        if not question.strip() or not answer.strip():
            return "❌ Question and Answer cannot be empty.", gr.update()

        try:
            current_fid = int(idx)
            start_fid, _, is_video = self.calculate_data_range(current_fid, data_type)
            end_fid = current_fid + 1 

            frames = self.current_data['episode_frames']
            dataset_name = self.current_data['dataset_name']
            episode_name = self.current_data['episode_name']
            
            saved_visual_path = None
            
            if not is_video:
                images_paths = []
                for fid in range(start_fid, end_fid):
                    img_p = ensure_image_on_disk(
                        dataset_name,
                        frames[fid]['exo_camera_key'],
                        episode_name,
                        fid,
                        frames[fid]['exo_image']
                    )
                    images_paths.append(img_p)
                saved_visual_path = images_paths[0] if len(images_paths) == 1 else images_paths
            else:
                imgs = [frames[fid]['exo_image'] for fid in range(start_fid, end_fid)]
                saved_visual_path = ensure_video_on_disk(
                    dataset_name,
                    frames[start_fid]['exo_camera_key'],
                    episode_name,
                    start_fid,
                    end_fid,
                    imgs,
                    self.fps
                )

            record = {
                "conversations": [{"from": "human", "value": question}, {"from": "gpt", "value": answer}],
                "meta": {
                    "dataset": dataset_name,
                    "episode": episode_name,
                    "frame_index": current_fid,
                    "task": frames[current_fid]['task'],
                    "sub_task": frames[current_fid]['sub_task']
                }
            }
            if is_video:
                record["video"] = saved_visual_path
            else:
                record["image"] = saved_visual_path

            append_QA_to_json(record)
            return f"✅ Saved Frame {current_fid}!", gr.update(value="", placeholder="Saved. Enter next...")
            
        except Exception as e:
            logging.error(f"Save failed: {e}")
            return f"❌ Error: {e}", gr.update()


# --- 4. 构建并启动 UI ---
def run_ui(datasets, task_ids):
    # 初始化生成器
    gen = episode_data_generator(datasets, task_ids)
    # 初始化逻辑控制器
    inspector = InspectorUI(gen)

    with gr.Blocks(title="AnnotatedLerobot Inspector") as demo:
        # Header
        header_md = gr.Markdown("## Click 'Start' to load data.")
        
        # Data Area
        with gr.Column(visible=False) as main_area:
            with gr.Row():
                with gr.Column(scale=2):
                    img_display = gr.Image(label="View", type="pil", interactive=False)
                    slider = gr.Slider(0, 100, value=0, step=1, label="Timeline")
                    with gr.Row():
                        prev_btn = gr.Button("<< Prev Frame")
                        next_btn = gr.Button("Next Frame >>")
                
                with gr.Column(scale=1):
                    info_md = gr.Markdown("Info")
                    plan_box = gr.Textbox(label="Planned Actions", lines=4, interactive=False)
                    data_type_dd = gr.Dropdown(
                        choices=["Single Image", "Image Sequence (4 frames)", "Video (2s)", "Video (Full Task)"],
                        value="Single Image", label="Save Type"
                    )
                    preview_lbl = gr.Label(show_label=False)

            gr.Markdown("---")
            
            # QA Area
            with gr.Row():
                with gr.Column():
                    q_input = gr.Textbox(label="Question", lines=2)
                    refresh_q = gr.Button("🎲 Random Template")
                with gr.Column():
                    a_input = gr.Textbox(label="Answer", lines=2)
            
            with gr.Row():
                save_btn = gr.Button("💾 Save QA Pair", variant="primary")
                status_msg = gr.Markdown("")

        gr.Markdown("---")
        # Footer Control
        load_next_btn = gr.Button("🚀 Start / Next Episode", variant="stop")

        # --- Event Wiring ---
        
        # 1. Load Next Episode
        load_next_btn.click(
            fn=inspector.load_next_episode,
            inputs=[],
            outputs=[header_md, slider, status_msg, save_btn, main_area]
        ).then( # Auto refresh image after load
            fn=inspector.get_frame_info,
            inputs=[slider],
            outputs=[img_display, info_md, plan_box, q_input, a_input]
        )

        # 2. Slider & Navigation
        slider.change(
            fn=inspector.get_frame_info,
            inputs=[slider],
            outputs=[img_display, info_md, plan_box, q_input, a_input]
        ).then(
            fn=inspector.update_preview_label,
            inputs=[slider, data_type_dd],
            outputs=[preview_lbl]
        )

        prev_btn.click(lambda x: max(0, x-1), inputs=[slider], outputs=[slider])
        next_btn.click(lambda x: x+1, inputs=[slider], outputs=[slider]) # Slider max will clip it

        # 3. Save Logic
        save_btn.click(
            fn=inspector.save_annotation,
            inputs=[slider, q_input, a_input, data_type_dd],
            outputs=[status_msg, a_input]
        )
        
        # 4. Helpers
        refresh_q.click(fn=inspector.get_random_question, outputs=[q_input])
        data_type_dd.change(fn=inspector.update_preview_label, inputs=[slider, data_type_dd], outputs=[preview_lbl])

    demo.launch(height=900, inbrowser=False, share=True)


if __name__ == "__main__":   
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset_paths", type=str, nargs='+', help="List of dataset paths.", required=True)
    argparser.add_argument("--json_output_path", type=str, help="Path to save output JSON.", required=True)
    argparser.add_argument("--data_output_dir", type=str, help="Directory to save visual data.", required=True)
    argparser.add_argument("--task_ids", type=str, nargs='+', required=True)
    argparser.add_argument("--templates_dir", type=str, default="evqa/templates")
    argparser.add_argument("--selection_file", type=str, default=None)
    args = argparser.parse_args()

    templates_dir = args.templates_dir
    data_output_dir = args.data_output_dir
    json_output_path = args.json_output_path # Fixed typo from json_output_dir
    selection_path = args.selection_file

    # 启动单例 UI
    run_ui(args.dataset_paths, args.task_ids)