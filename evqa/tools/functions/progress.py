import logging
import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union
from core.utils.common import image_phash_sim


def r(frames: List[Dict], tau=None) -> Union[List[float], float]:
    """
    Compute the r(tau) values for the progress function based on the provided frames.
    Args:
        frames (List[Dict]): A list of frame dictionaries, each containing:
            - 'exo_image': PIL.Image, image from the third-person camera
        tau (Optional[int]): An optional index to return a specific r(tau) value.
    Returns:
        Union[List[float], float]: A list of r(tau) values or a single r(tau) value if tau is specified.
    """
    results = []
    if len(frames) == 1:
        results.append(0.0)
        logging.warning("When computing r(tau) for progress function: only one frame available, returning 0.0")
    else:
        dphi = []
        for i in range(1, len(frames)):
            dphi.append(1 + 0.5 - image_phash_sim(frames[i - 1]['exo_image'], frames[i]['exo_image']))
        total_dphi = sum(dphi)
        if total_dphi == 0:
            logging.warning("When computing r(tau) for progress function: all frames are identical, returning 0.0 for all r(tau)")
        for i in range(len(frames)):
            if i == 0:
                results.append(0.0)
            else:
                results.append(dphi[i - 1] / total_dphi if total_dphi > 0 else 0.0)
    if tau is not None:
        return results[tau]
    return results

def p(frames: List[Dict], tau=None) -> Union[List[float], float]:
    """
    Compute the p(tau) values for the progress function based on the provided frames.
    Args:
        frames (List[Dict]): A list of frame dictionaries, each containing:
            - 'exo_image': PIL.Image, image from the third-person camera
            - 'sub_task': str, sub-task description
        tau (Optional[int]): An optional index to return a specific p(tau) value.
    Returns:
        Union[List[float], float]: A list of p(tau) values or a single p(tau) value if tau is specified.
    """
    assert len(frames) > 0, "When computing p(tau) for progress function: frames list is empty"
    splits = []
    task_start = 0
    for task_end in range(1, len(frames)):
        if frames[task_end]['sub_task'] != frames[task_end - 1]['sub_task']:
            splits.append((task_start, task_end, frames[task_end - 1]['sub_task']))
            task_start = task_end
    splits.append((task_start, len(frames), frames[-1]['sub_task']))

    valid_task_masks = []
    T = K = 0
    for s, e, task in splits:
        if task and task.lower() not in ['done', 'done.', 'not done', 'not done.']:
            valid_task_masks.append(True)
            K += 1
            T += (e - s)
        else:
            valid_task_masks.append(False)
    if T == 0:
        logging.warning("When computing p(tau) for progress function: no valid sub-tasks found, all p(tau) will be 0.0")

    dp = []
    for i, (s, e, _) in enumerate(splits):
        if not valid_task_masks[i]:
            dp.extend([0.0] * (e - s))
        else:
            w_value = np.clip(K * (e - s) / T if T > 0 else 0.0, 0.75, 1.25)
            r_values = r(frames[s:e])
            dp.extend([w_value * r_val for r_val in r_values])
    
    total_dp = sum(dp)
    results = [0.0]
    for i in range(1, len(frames)):
        results.append(sum(dp[:i]) / total_dp if total_dp > 0 else 0.0)
    
    if tau is not None:
        return results[tau]
    return results