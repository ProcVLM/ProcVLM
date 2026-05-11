import torch
import numpy as np
from PIL import Image
from scipy import stats
# from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import RANSACRegressor, LinearRegression
from typing import List, Optional, Tuple, Dict, Any, Union
from core.utils.common import image_phash_sim

sqrt2 = np.sqrt(2.0)
@torch.no_grad()
def DS(box: List[int], F_boxes: List[List[int]], alpha: float = 0.75, beta: float = 0.25) -> float:
    """
    Compute the estimated Distance Score between a given box and all boxes from a certain frame F.

    The formula for the Distance Score is defined as below:
        DS(box, F) = Min_{b in F} BOX_DIST(box, b)
    where BOX_DIST is the weighted sum of box center distance (using Euclidean distance) and box area difference (using IoU).
        BOX_DIST(box1, box2) = alpha * || center(box1) - center(box2) ||_2 / sqrt(2) + beta * (1 - IoU(box1, box2))
    As alpha + beta = 1, the BOX_DIST and DS is always in the range of [0, 1].

    Args:
        box (List[int]): A list of four integers representing the coordinates of the box [x1, y1, x2, y2].
        F_boxes (List[List[int]]): A list of boxes, each represented by a list of four integers.
        alpha (float): Weight for the box center distance component.
        beta (float): Weight for the box area difference component.

    Returns:
        float: The distance score between the given box and each box in F_boxes.
    """
    box = torch.tensor(box, dtype=torch.float32, device="cuda")
    F_boxes = torch.tensor(F_boxes, dtype=torch.float32, device="cuda")

    # Center of the input box
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    # Centers of reference boxes
    Fx = (F_boxes[:, 0] + F_boxes[:, 2]) / 2
    Fy = (F_boxes[:, 1] + F_boxes[:, 3]) / 2
    dists = torch.sqrt((Fx - cx) ** 2 + (Fy - cy) ** 2)

    # Intersection
    inter_x1 = torch.max(box[0], F_boxes[:, 0])
    inter_y1 = torch.max(box[1], F_boxes[:, 1])
    inter_x2 = torch.min(box[2], F_boxes[:, 2])
    inter_y2 = torch.min(box[3], F_boxes[:, 3])
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h

    # IoU
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    F_area = (F_boxes[:, 2] - F_boxes[:, 0]) * (F_boxes[:, 3] - F_boxes[:, 1])
    union_area = box_area + F_area - inter_area
    ious = inter_area / torch.clamp(union_area, min=1e-8)

    # Distance score
    box_dists = alpha * dists / sqrt2 + beta * (1 - ious)
    return float(torch.min(box_dists).item())


@torch.no_grad()
def DS_vec(boxes: List[List[int]], F_boxes: List[List[int]], alpha: float = 0.75, beta: float = 0.25) -> np.ndarray:
    """
    Vectorized DS for multiple boxes against a set of reference boxes.

    Args:
        boxes (List[List[int]]): List of boxes. Shape: (N, 4)
        F_boxes (List[List[int]]): Reference boxes. Shape: (M, 4)
        alpha (float): weight for normalized Euclidean distance
        beta (float): weight for IoU

    Returns:
        np.ndarray: Minimum distance score for each box. Shape: (N,)
    """
    boxes = torch.tensor(boxes, dtype=torch.float32, device="cuda")
    F_boxes = torch.tensor(F_boxes, dtype=torch.float32, device="cuda")

    # Centers
    cxs = (boxes[:, 0] + boxes[:, 2]) / 2
    cys = (boxes[:, 1] + boxes[:, 3]) / 2
    Fxs = (F_boxes[:, 0] + F_boxes[:, 2]) / 2
    Fys = (F_boxes[:, 1] + F_boxes[:, 3]) / 2

    # Pairwise distances: shape (N, M)
    dx = cxs[:, None] - Fxs[None, :]
    dy = cys[:, None] - Fys[None, :]
    dists = torch.sqrt(dx**2 + dy**2)

    # IoU: shape (N, M)
    x1 = torch.max(boxes[:, None, 0], F_boxes[None, :, 0])
    y1 = torch.max(boxes[:, None, 1], F_boxes[None, :, 1])
    x2 = torch.min(boxes[:, None, 2], F_boxes[None, :, 2])
    y2 = torch.min(boxes[:, None, 3], F_boxes[None, :, 3])
    inter_w = torch.clamp(x2 - x1, min=0)
    inter_h = torch.clamp(y2 - y1, min=0)
    inter_area = inter_w * inter_h

    box_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])    # (N,)
    F_area = (F_boxes[:, 2] - F_boxes[:, 0]) * (F_boxes[:, 3] - F_boxes[:, 1])  # (M,)
    union_area = box_area[:, None] + F_area[None, :] - inter_area
    ious = inter_area / torch.clamp(union_area, min=1e-8)   # (N,M)

    box_dists = alpha * dists / sqrt2 + beta * (1 - ious)   # (N,M)
    return torch.min(box_dists, dim=1).values.cpu().numpy() # (N,)


def DIoU_loss_vec(boxes: List[List[int]], F_boxes: List[List[int]], alpha: float=0.75, beta: float=0.25) -> np.array:
    """
    Vectorized computation of Distance IoU loss for multiple boxes against a set of boxes F.
        DIoU_loss(box1, box2) = alpha * ||center(box1) - center(box2)||_2 / ||(MinX, MinY) - (MaxX, MaxY)||_2 + beta * (1 - IoU(box1, box2))

    Args:
        boxes (List[List[int]]): A list of boxes, each represented by a list of four integers.
        F_boxes (List[List[int]]): A list of boxes, each represented by a list of four integers.
        alpha (float): Weight for the box center distance component.
        beta (float): Weight for the box area difference component.

    Returns:
        List[float]: A list of Distance IoU loss values for each box in boxes against F_boxes. If the shape of boxes is (N, 4) and F_boxes is (M, 4), the output will be a list of length N.
    """
    boxes = np.array(boxes)
    F_boxes = np.array(F_boxes)

    # Compute Euclidean distances between box centers
    cxs = (boxes[:, 0] + boxes[:, 2]) / 2
    cys = (boxes[:, 1] + boxes[:, 3]) / 2
    Fxs = (F_boxes[:, 0] + F_boxes[:, 2]) / 2
    Fys = (F_boxes[:, 1] + F_boxes[:, 3]) / 2

    dx = cxs[:, None] - Fxs[None, :]
    dy = cys[:, None] - Fys[None, :]
    cds = np.sqrt(dx**2 + dy**2) # (N,M)

    MinX = np.minimum(boxes[:, None, 0], F_boxes[None, :, 0]) # (N,M)
    MinY = np.minimum(boxes[:, None, 1], F_boxes[None, :, 1])
    MaxX = np.maximum(boxes[:, None, 2], F_boxes[None, :, 2])
    MaxY = np.maximum(boxes[:, None, 3], F_boxes[None, :, 3])
    ads = np.sqrt((MaxX - MinX)**2 + (MaxY - MinY)**2) # (N,M)
    dists = cds / np.clip(ads, 1e-8, None) # (N,M)

    # Compute IoU between each box in boxes and each box in F_boxes
    x1 = np.maximum(boxes[:, None, 0], F_boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], F_boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], F_boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], F_boxes[None, :, 3])

    inter_w = np.clip(x2 - x1, 0, None)
    inter_h = np.clip(y2 - y1, 0, None)
    inter_area = inter_w * inter_h

    box_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])   # (N,)
    F_area   = (F_boxes[:, 2] - F_boxes[:, 0]) * (F_boxes[:, 3] - F_boxes[:, 1])  # (M,)

    union_area = box_area[:, None] + F_area[None, :] - inter_area
    ious = inter_area / np.clip(union_area, 1e-8, None)  # (N,M)

    box_dists = alpha * dists + beta * (1 - ious)  # (N,M)
    return np.min(box_dists, axis=1)  # (N,)


@torch.no_grad()
def D_mat(boxes: np.ndarray, Fs: np.ndarray, padding_mask: np.ndarray, alpha: float = 0.75, beta: float = 0.25) -> np.ndarray:
    """
    Compute distance matrix between boxes and a set of frame-wise reference boxes.

    Args:
        boxes (np.ndarray): Boxes. Shape: (N, 4)
        Fs (np.ndarray): Reference boxes. Shape: (M, B, 4)
        padding_mask (np.ndarray): Mask for valid entries. Shape: (M, B)
        alpha (float): weight for normalized distance
        beta (float): weight for IoU

    Returns:
        np.ndarray: Distance matrix. Shape: (N, M, B)
    """
    boxes = torch.tensor(boxes, dtype=torch.float32, device="cuda")
    Fs = torch.tensor(Fs, dtype=torch.float32, device="cuda")
    padding_mask = torch.tensor(padding_mask, device="cuda")

    # Centers
    cxs = (boxes[:, 0] + boxes[:, 2]) / 2
    cys = (boxes[:, 1] + boxes[:, 3]) / 2
    Fxs = (Fs[:, :, 0] + Fs[:, :, 2]) / 2
    Fys = (Fs[:, :, 1] + Fs[:, :, 3]) / 2

    # Pairwise distances: shape (N, M, B)
    dx = cxs[:, None, None] - Fxs[None, :, :]
    dy = cys[:, None, None] - Fys[None, :, :]
    dists = torch.sqrt(dx**2 + dy**2)
    dists = torch.where(padding_mask[None, :, :], dists, torch.tensor(float("inf"), device="cuda"))

    del cxs, cys, Fxs, Fys, dx, dy
    torch.cuda.empty_cache()

    # IoU
    boxes_exp = boxes[:, None, None, :]  # (N,1,1,4)
    Fs_exp = Fs[None, :, :, :]           # (1,M,B,4)

    x1 = torch.max(boxes_exp[..., 0], Fs_exp[..., 0])
    y1 = torch.max(boxes_exp[..., 1], Fs_exp[..., 1])
    x2 = torch.min(boxes_exp[..., 2], Fs_exp[..., 2])
    y2 = torch.min(boxes_exp[..., 3], Fs_exp[..., 3])
    inter_w = torch.clamp(x2 - x1, min=0)
    inter_h = torch.clamp(y2 - y1, min=0)
    inter_area = inter_w * inter_h

    del x1, y1, x2, y2, inter_w, inter_h
    torch.cuda.empty_cache()

    box_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # (N,)
    Fs_area = (Fs[..., 2] - Fs[..., 0]) * (Fs[..., 3] - Fs[..., 1])       # (M,B)

    union_area = box_area[:, None, None] + Fs_area[None, :, :] - inter_area  # (N,M,B)
    ious = inter_area / torch.clamp(union_area, min=1e-8)   # (N,M,B)
    ious = torch.where(padding_mask[None, :, :], ious, torch.zeros_like(ious))  # (N,M,B)

    del boxes_exp, Fs_exp, inter_area, box_area, Fs_area, union_area
    torch.cuda.empty_cache()

    # Distance score: shape (N,M,B)
    box_dists = alpha * dists / sqrt2 + beta * (1 - ious)   # (N,M,B)
    box_dists = torch.where(padding_mask[None, :, :], box_dists, torch.tensor(float("inf"), device="cuda")) # (N,M,B)
    return box_dists.cpu().numpy()  # (N,M,B)


@torch.no_grad()
def DS_mat(boxes: np.ndarray, Fs: np.ndarray, padding_mask: np.ndarray, alpha: float = 0.75, beta: float = 0.25) -> np.ndarray:
    """
    Compute minimum distance score along the last axis of D_mat.

    Args:
        boxes (np.ndarray): Boxes. Shape: (N, 4)
        Fs (np.ndarray): Reference boxes. Shape: (M, B, 4)
        padding_mask (np.ndarray): Mask. Shape: (M, B)
        alpha (float): weight for distance
        beta (float): weight for IoU

    Returns:
        np.ndarray: Minimum distance matrix. Shape: (N, M)
    """
    return np.min(D_mat(boxes, Fs, padding_mask, alpha, beta), axis=2)  # (N,M)


def Vel_vec(boxes: List[List[int]], all_frame_boxes: List[List[int]], cur_idx: int) -> List[float]:
    """
    Vectorized computation of velocity scores for multiple boxes based on their movement across frames.

    Args:
        boxes (List[List[int]]): A list of boxes at the current frame, each represented by a list of four integers.
        all_frame_boxes (List[List[int]]): A list of all boxes across frames, where each box is represented by a list of four integers.
        cur_idx (int): The index of the current frame in the sequence.

    Returns:
        List[float]: A list of velocity scores for each box in boxes. If the shape of boxes is (N, 4), the output will be a list of length N.
    """
    if boxes is None or len(boxes) == 0:
        return []

    other_frame_boxes = all_frame_boxes[:cur_idx] + all_frame_boxes[cur_idx+1:]
    M = len(other_frame_boxes)
    assert M > 0, "No other boxes to compare for velocity computation."

    boxes = np.array(boxes)  # (N,4)

    # get padded other_frame_boxes and mask
    max_b = max(len(f) for f in other_frame_boxes)
    padded_other_boxes = np.zeros((len(other_frame_boxes), max_b, 4), dtype=float)
    mask = np.zeros((len(other_frame_boxes), max_b), dtype=bool)
    for i, f in enumerate(other_frame_boxes):
        if len(f) > 0:
            padded_other_boxes[i, :len(f)] = f
            mask[i, :len(f)] = True
    # padded_other_boxes: (M,max_b,4), mask: (M,max_b)

    res = DS_mat(boxes, padded_other_boxes, mask)  # (N,M)
    # ts = np.concatenate([np.arange(1, cur_idx + 1)[::-1], np.arange(1, M - cur_idx + 1)])
    # vels = (res / ts[None, :]).mean(axis=1)  # (N,)
    vels = res.mean(axis=1)  # (N,)
    return vels.tolist()


def rulebased_remove_upperleft_outlier(bboxes: List[Dict[str, Any]]) -> List[int]:
    if len(bboxes) <= 2:
        return range(len(bboxes))
    def is_upperleft(box, thresh=0.5):
        _, _, x2, y2 = box['bbox_2d']
        return x2 < thresh and y2 < thresh
    def too_small(box, min_len=0.05):
        x1, y1, x2, y2 = box['bbox_2d']
        max_len = max(x2 - x1, y2 - y1)
        return max_len < min_len
    if len(bboxes) >= 8:
        return [i for i, bbox in enumerate(bboxes) if not is_upperleft(bbox) and not too_small(bbox)]
    if len(bboxes) >= 5:
        return [i for i, bbox in enumerate(bboxes) if not is_upperleft(bbox)]
    if all(is_upperleft(bbox) for bbox in bboxes):
        return []
    if sum(1 for bbox in bboxes if is_upperleft(bbox) and too_small(bbox)) >= 2:
        return [i for i, bbox in enumerate(bboxes) if not (is_upperleft(bbox) and too_small(bbox))]
    if sum(1 for bbox in bboxes if is_upperleft(bbox)) >= 3:
        return [i for i, bbox in enumerate(bboxes) if not is_upperleft(bbox)]
    return range(len(bboxes))

class GammaDistribution:
    eps = 1e-6
    def __init__(self, vels: List[float], min_pos_count=1):
        assert all(v >= 0 for v in vels), f"Negative velocity found, report bug!"
        pos_cnt = sum(v > 0 for v in vels)
        if pos_cnt < min_pos_count or np.std(vels) < self.eps:
            self.all_inliers = True
            return
        self.all_inliers = False
        vels = [v + self.eps for v in vels] # gamma distribution requires positive values
        self.a, self.loc_g, self.scale_g = stats.gamma.fit(vels, floc=0)

    def gamma_pvalue(self, v):
        return stats.gamma.sf(v, self.a, loc=self.loc_g, scale=self.scale_g)
    
    def is_outlier(self, v):
        if self.all_inliers:
            return False
        p = self.gamma_pvalue(v)
        return p < 0.1 # 90% confidence interval

class LogNormalDistribution:
    eps = 1e-6
    
    def __init__(self, vels: List[float], min_pos_count=1):
        assert all(v >= 0 for v in vels), f"Negative velocity found, report bug!"
        pos_cnt = sum(v > 0 for v in vels)
        if pos_cnt < min_pos_count or np.std(vels) < self.eps:
            self.all_inliers = True
            return
        
        self.all_inliers = False
        vels = [v + self.eps for v in vels]  # avoid log(0)
        
        # lognorm parameterization: shape (sigma), loc, scale (exp(mu))
        # set loc=0 to ensure positive values
        self.sigma, _, self.scale = stats.lognorm.fit(vels, floc=0)

    def lognorm_pvalue(self, v):
        if v <= 0:  # lognormal is only defined for positive numbers
            return 0.0
        return stats.lognorm.sf(v, self.sigma, loc=0, scale=self.scale)
    
    def is_outlier(self, v):
        if self.all_inliers:
            return False
        p = self.lognorm_pvalue(v)
        return p < 0.05  # 95% confidence interval

def split_episode_by_task(frames: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    if len(frames) == 0:
        return []
    splits = []
    ck_set = set(item['camera_key'] for item in frames)
    for ck in ck_set:
        ck_frames = [f for f in frames if f['camera_key'] == ck]
        ck_frames = sorted(ck_frames, key=lambda x: x['frame_id'])
        episode = []
        last_task = None
        for f in ck_frames:
            if last_task is None or f['task_desc'] != last_task:
                if len(episode) > 0:
                    splits.append(episode)
                episode = [f]
                last_task = f['task_desc']
            else:
                episode.append(f)
        if len(episode) > 0:
            splits.append(episode)
    return splits

def detect_outliers_ransac(x, y, residual_threshold=None, min_samples=2, random_state=42):
    """
    Detect outliers in a sequence using RANSAC regression.

    Args:
        x (array-like): 1D or 2D array of input features. Shape (n,) or (n, 1).
        y (array-like): 1D array of target values. Shape (n,).
        residual_threshold (float, optional): The maximum residual for a data point to be classified as an inlier.
                                              If None, it will be set to 1.5 times the standard deviation of y.
        min_samples (int): The minimum number of data points to fit the model.
        random_state (int): Random seed for reproducibility. 

    Returns:
        outliers (np.array): Indices of the detected outliers in the input sequence.
        inlier_mask (np.array): A boolean array where True indicates an inlier.
        outlier_mask (np.array): A boolean array where True indicates an outlier.
        residuals (np.array): Absolute residuals (distance to fitted line) for each data point.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # Ensure x has shape (n_samples, 1)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n = len(y)
    if n < min_samples:
        return (
            np.array([], dtype=int),
            np.ones(n, dtype=bool),
            np.zeros(n, dtype=bool),
            np.zeros(n, dtype=float),
        )
    if residual_threshold is None:
        residual_threshold = 1.5 * np.std(y)
    model = RANSACRegressor(
        estimator=LinearRegression(),
        min_samples=min_samples,
        residual_threshold=residual_threshold,
        random_state=random_state
    )
    model.fit(x, y)
    y_pred = model.predict(x)
    residuals = np.abs(y - y_pred)
    inlier_mask = model.inlier_mask_
    outlier_mask = ~inlier_mask
    outliers = np.where(outlier_mask)[0]
    return outliers, inlier_mask, outlier_mask, residuals

def interpolate_boxes(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Interpolate missing bounding boxes in a sequence of frames.

    Args:
        frames (List[Dict[str, Any]]): A list of frames, each represented by a json dictionary like:   
            {
                "frame_id": r_idx,
                "camera_key": cam_key,
                "task_desc": task,
                "bboxes": [obj1, obj2, ...],
                ... other keys ...
            }
        Each object in "bboxes" is a dictionary with at least the following keys:
            {
                "label": str,
                "bbox_2d": [x1, y1, x2, y2],  # regular coordinates to [0, 1]
                "score": float,
            }

    Returns:
        List[Dict[str, Any]]: The input list with missing bounding boxes interpolated.
    """
    def do_name_check(label: str, boxes: List[Dict[str, Any]]) -> bool:
        for b in boxes:
            if b['label'] == label:
                return False
        return True

    def do_dist_check(box, boxes, threshold=0.6) -> bool:
        if len(boxes) == 0:
            return True
        dists = DIoU_loss_vec([b['bbox_2d'] for b in boxes], [box], 0.25, 0.75)
        return dists.min() > threshold

    def crop_box(image: Image.Image, box: List[int]) -> Image.Image:
        w, h = image.size
        x1, y1, x2, y2 = box
        return image.crop((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))).resize((128, 128), Image.Resampling.LANCZOS)
    
    def check_box_sim(box1, img1, box2, img2, sim_thresh=0.7) -> bool:
        """0.7 means at most 20 different bits in phash (64 bits) is allowed"""
        if img1 is None or img2 is None:
            return False
        sim = image_phash_sim(crop_box(img1, box1), crop_box(img2, box2))
        if sim >= sim_thresh:
            return True
        return False

    def cluster_worker(frames: List[Dict[str, Any]], future_size: int=90, max_diff: float=0.1) -> List[Dict[str, Any]]:
        N = len(frames)
        if N == 0:
            for f in frames:
                del f['image']
            return frames
        B = max(len(f['bboxes']) for f in frames)
        if B == 0:
            for f in frames:
                del f['image']
            return frames
        
        # should have a distance matrix of shape (N, B, N, B)
        all_boxes = [f['bboxes'] for f in frames]
        padded_boxes = np.zeros((N, B, 4), dtype=float)
        mask = np.zeros((N, B), dtype=bool)
        for i, bbs in enumerate(all_boxes):
            if len(bbs) > 0:
                padded_boxes[i, :len(bbs)] = [bb['bbox_2d'] for bb in bbs]
                mask[i, :len(bbs)] = True
        dists = D_mat(padded_boxes.reshape(-1, 4), padded_boxes, mask, alpha=0.5, beta=0.5)  # (N*B, N, B)
        dists = dists.reshape(N, B, N, B)
        dists = np.where(mask[None, None, :, :], dists, 1e6) # (N,B,N,B)
        dists = np.where(mask[:, :, None, None], dists, 1e6) # (N,B,N,B)
        dists = np.where(np.eye(N, dtype=bool)[:, None, :, None], 1e6, dists) # (N,B,N,B)

        # should have Maximal Bipartite Matching of shape (N, B, N), where (i,j,k)=v means considering frame i and k, the j-th box matches to box v, if v=-1 means no match
        MBM = np.full((N, B, N), -1, dtype=int)
        # should have a matrix of shape (N, B), where (i, j)=(k, v) means the next hop is frame k, box v
        next_hop = np.full((N, B, 2), -1, dtype=int) # (N,B,2)
        for i in range(N):
            for k in range(i + 1, N):
                cost_matrix = dists[i, :, k, :]  # (B,B)
                # row_ind, col_ind = linear_sum_assignment(cost_matrix)
                # for r, c in zip(row_ind, col_ind):
                for r, c in np.ndindex((B, B)):
                    if cost_matrix[r, c] < max_diff and mask[i, r] and mask[k, c]: # valid match
                        MBM[i, r, k] = c # single direction
                        # MBM[k, c, i] = r
            for j in range(B):
                if MBM[i, j].max() == -1:
                    continue
                match_dists = []
                match_ids = []
                for k in range(i + 1, N):
                    if MBM[i, j, k] != -1:
                        match_dists.append(dists[i, j, k, MBM[i, j, k]])
                        match_ids.append(k)
                # the distribution of match distance of a single box should be roughly linear,
                # using RANSAC to filter out outliers
                _, _, outlier_mask, _ = detect_outliers_ransac(match_ids, match_dists, residual_threshold=1)
                for idx, is_outlier in zip(match_ids, outlier_mask):
                    if is_outlier:
                        MBM[i, j, idx] = -1
                # find the next hop
                for k in range(i + 1, min(i + 1 + future_size, N)):
                    if MBM[i, j, k] != -1:
                        next_hop[i, j] = [k, MBM[i, j, k]]
                        break
                # argmin_residual = np.argmin(residuals[:future_size])
                # if residuals[argmin_residual] < 0.1 and MBM[i, j, i + 1 + argmin_residual] != -1:
                #     next_hop[i, j] = [i + 1 + argmin_residual, MBM[i, j, i + 1 + argmin_residual]]

        # interpolate box between frames
        new_frames = [f.copy() for f in frames] 
        for f in new_frames:
            del f['image']
            f['bboxes'] = f['bboxes'].copy()
        for i in range(N):
            frame = frames[i]
            for j in range(B):
                if not mask[i, j]:
                    continue
                ni, nj = next_hop[i, j]
                if ni == -1 or (ni - i) <= 1:
                    continue
                # interpolate box from frame i to ni
                box_i = padded_boxes[i, j]
                box_ni = padded_boxes[ni, nj]
                for inter_i in range(i + 1, ni):
                    ratio = (inter_i - i) / (ni - i)
                    lbl = frame['bboxes'][j]['label']
                    box = (box_i * (1 - ratio) + box_ni * ratio).tolist()
                    if (do_name_check(lbl, new_frames[inter_i]['bboxes'])
                        and do_dist_check(box, new_frames[inter_i]['bboxes']) 
                        and check_box_sim(box_i, frame['image'], box, frames[inter_i]['image'])
                        and check_box_sim(box_ni, frames[ni]['image'], box, frames[inter_i]['image'])):
                        new_bbox = {
                            'bbox_2d': box,
                            'label': lbl,
                            'score': 'interpolated',
                        }
                        # print(f"Interpolate box at frame {inter_i}: {new_bbox['label']} {new_bbox['bbox_2d']}")
                        new_frames[inter_i]['bboxes'].append(new_bbox)
        return new_frames

    # main function body
    result = []
    splits = split_episode_by_task(frames)
    for spl in splits:
        new_jl = cluster_worker(spl)
        result.extend(new_jl)
    return result