import re
import json_repair
import fasttext
import numpy as np
import itertools
from functools import lru_cache
from typing import Tuple, List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr


############################
# FastText
############################

@lru_cache(maxsize=1)
def _load_fasttext(fasttext_bin_path="assets/fasttext/cc.en.300.bin"):
    return fasttext.load_model(fasttext_bin_path)

def _tokenize_en(text: str):
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)

STOPWORDS = set([
    "i","you","he","she","it","we","they","me","him","her","us","them",
    "a","an","the","is","are","was","were","be","been",
    "and","or","but","if","then","so",
])


############################
# Label similarity
############################

def sentence_embedding(text: str):
    model = _load_fasttext()
    tokens = [w for w in _tokenize_en(text) if w not in STOPWORDS]
    if not tokens:
        return None
    vecs = np.stack([model.get_word_vector(w) for w in tokens])
    return vecs.mean(axis=0)

def label_similarity(pred_label: str, gt_label: str):
    if pred_label.strip().lower() == gt_label.strip().lower():
        return 1.0
    v1 = sentence_embedding(pred_label)
    v2 = sentence_embedding(gt_label)
    if v1 is None or v2 is None:
        return 0.0
    sim = cosine_similarity(v1.reshape(1,-1), v2.reshape(1,-1))[0,0]
    return float(np.clip(sim,0,1))


############################
# Progress extraction
############################

def extract_progress_value(txt: str) -> float | None:
    """ Extract a numeric progress value (0-100) from the text. 
    Normalize to percentage if it's in [0,1]. 
    If no valid number found, return None. """
    number_pattern = r"[-+]?(?:\d+\.\d+|\d+|\.\d+)"
    def parse(segment: str):
        m = re.search(number_pattern, segment)
        if not m:
            return None
        val = float(m.group(0))

        # followed by %
        if m.end() < len(segment) and segment[m.end()] == "%":
            return val
        # otherwise automatically determine the unit
        return val * 100 if val <= 1 else val

    # first look for <progress> tags
    tag = re.search(r"<progress>(.*?)</progress>", txt, re.IGNORECASE | re.DOTALL)
    if tag:
        res = parse(tag.group(1))
        if res is not None:
            return res
    # if no tags, parse the whole text
    return parse(txt)


############################
# Episode extraction
############################

def extract_episode_token(item: Dict) -> str | None:
    path = None
    if "image" in item:
        path = item["image"]
        if isinstance(path,list):
            path = path[0]
    if "video" in item:
        path = item["video"]
        if isinstance(path,list):
            path = path[0]
    if path is None:
        return None

    m = re.search(r"(episode_\d+)", path)
    if m:
        return m.group(1)
    return None


############################
# Segmentation
############################

SEGMENTATION_BOUNDARY_SIGMA = 5
def eval_action_segmentation(pred_txt, gt_txt, sigma=SEGMENTATION_BOUNDARY_SIGMA):
    def _extract_boundaries(segments):
        boundaries = []
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            duration = []
            for v in seg.values():
                if isinstance(v, (int, float)):
                    duration.append(float(v))
            if len(duration) >= 2:
                start = min(duration)
                end = max(duration)
                boundaries.extend([start, end])

        if not boundaries:
            return []

        # remove duplicate boundaries from adjacent segments such as end_i == start_{i+1}
        return sorted(set(boundaries))

    try:
        gt = json_repair.loads(gt_txt)
        pred = json_repair.loads(pred_txt)
    except Exception:
        return {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "matched_count": 0,
            "matched_abs_error_sum": 0.0,
            "f1": 0.0,
            "mae": 0.0,
        }

    if not isinstance(gt, list) or not isinstance(pred, list):
        return {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "matched_count": 0,
            "matched_abs_error_sum": 0.0,
            "f1": 0.0,
            "mae": 0.0,
        }

    gt_boundaries = _extract_boundaries(gt)
    pred_boundaries = _extract_boundaries(pred)

    def _sequence_length(boundaries):
        if len(boundaries) < 2:
            return 0.0
        return float(max(boundaries) - min(boundaries))

    seq_len = _sequence_length(gt_boundaries)
    if seq_len <= 0:
        seq_len = _sequence_length(pred_boundaries)
    delta = (float(sigma) / 100.0) * seq_len

    num_gt = len(gt_boundaries)
    num_pred = len(pred_boundaries)

    if num_gt == 0 and num_pred == 0:
        return {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "matched_count": 0,
            "matched_abs_error_sum": 0.0,
            "f1": 1.0,
            "mae": 0.0,
        }

    if num_gt == 0:
        return {
            "tp": 0,
            "fp": num_pred,
            "fn": 0,
            "matched_count": 0,
            "matched_abs_error_sum": 0.0,
            "f1": 0.0,
            "mae": 0.0,
        }

    if num_pred == 0:
        return {
            "tp": 0,
            "fp": 0,
            "fn": num_gt,
            "matched_count": 0,
            "matched_abs_error_sum": 0.0,
            "f1": 0.0,
            "mae": 0.0,
        }

    cost_matrix = np.zeros((num_gt, num_pred), dtype=float)
    for i, gt_t in enumerate(gt_boundaries):
        for j, pred_t in enumerate(pred_boundaries):
            cost_matrix[i, j] = abs(gt_t - pred_t)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_abs_errors = []
    for i, j in zip(row_ind, col_ind):
        err = cost_matrix[i, j]
        if err <= delta:
            matched_abs_errors.append(float(err))

    tp = len(matched_abs_errors)
    fp = num_pred - tp
    fn = num_gt - tp

    denom = 2 * tp + fp + fn
    f1 = (2 * tp / denom) if denom > 0 else 0.0
    mae = float(np.mean(matched_abs_errors)) if matched_abs_errors else 0.0

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "matched_count": int(tp),
        "matched_abs_error_sum": float(np.sum(matched_abs_errors)),
        "f1": float(f1),
        "mae": mae,
    }


############################
# Planning
############################

def split_actions(text: str):
    text = text.lower()
    # normalize separators
    text = text.replace("\r", "\n")
    # remove numbering like "1. action"
    text = re.sub(r"\b\d+\.\s*", "", text)
    # split on newline or period or semicolon
    parts = re.split(r"[.\n;]+", text)
    # strip and remove empty
    return [p.strip() for p in parts if p.strip()]

def eval_action_description_or_prediction(pred_txt, gt_txt):
    pred = split_actions(pred_txt)
    gt   = split_actions(gt_txt)
    if not pred or not gt:
        return 0.0

    # similarity matrix
    sim_matrix = np.zeros((len(pred), len(gt)))
    for i,p in enumerate(pred):
        for j,g in enumerate(gt):
            sim_matrix[i,j] = label_similarity(p,g)

    # Hungarian matching (maximize similarity)
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)
    matched_scores = sim_matrix[row_ind, col_ind]
    score_sum = matched_scores.sum()

    precision = score_sum / len(pred)
    recall    = score_sum / len(gt)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return float(f1)


############################
# Concordance Correlation Coefficient
############################

def compute_ccc(pred_txts, gt_txts, episodes):
    epi_data = {}
    for p, g, e in zip(pred_txts, gt_txts, episodes):
        pv = extract_progress_value(p)
        gv = extract_progress_value(g)
        
        if pv is None or gv is None:
            continue

        pv = max(0, min(100, pv))
        gv = max(0, min(100, gv))
        epi_data.setdefault(e, []).append((gv, pv))  # (GT, Pred)

    # compute CCC for each episode and average
    scores = []
    for seq in epi_data.values():
        if len(seq) < 2:
            continue

        gts = np.array([x[0] for x in seq], dtype=float)
        preds = np.array([x[1] for x in seq], dtype=float)

        mean_gt = gts.mean()
        mean_pred = preds.mean()

        var_gt = gts.var()
        var_pred = preds.var()

        # if either variance is too small, CCC is not well-defined, assign 0 (no concordance)
        if var_pred < 1e-8 or var_gt < 1e-8:
            scores.append(0.0)
            continue

        corr = np.corrcoef(gts, preds)[0, 1]
        if np.isnan(corr):
            scores.append(0.0)
            continue

        std_gt = np.sqrt(var_gt)
        std_pred = np.sqrt(var_pred)

        # formula: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
        ccc = (
            2 * corr * std_pred * std_gt
            / (var_pred + var_gt + (mean_pred - mean_gt) ** 2 + 1e-8)
        )

        scores.append(float(ccc))

    return float(np.mean(scores)) if scores else 0.0


############################
# Unique Bin Ratio
############################

def compute_ubr(pred_txts, gt_txts, episodes, bin_size=5):
    epi_data = {}

    for p, g, e in zip(pred_txts, gt_txts, episodes):
        pv = extract_progress_value(p)
        gv = extract_progress_value(g)

        if pv is None or gv is None:
            continue

        pv = max(0, min(100, pv))
        gv = max(0, min(100, gv))

        # exclude trivial static boundary cases
        if gv <= 0 or gv >= 100:
            continue

        epi_data.setdefault(e, []).append((gv, pv))  # (GT, Pred)

    scores = []

    for seq in epi_data.values():
        if len(seq) < 2:
            continue

        gts = np.array([x[0] for x in seq], dtype=float)
        preds = np.array([x[1] for x in seq], dtype=float)

        # quantize with bin size
        gt_bins = np.round(gts / bin_size) * bin_size
        pred_bins = np.round(preds / bin_size) * bin_size

        num_gt_bins = len(np.unique(gt_bins))
        num_pred_bins = len(np.unique(pred_bins))

        if num_gt_bins == 0:
            continue

        ubr = num_pred_bins / num_gt_bins
        scores.append(float(ubr))

    return float(np.mean(scores)) if scores else 0.0


############################
# Effective Progress Resolution
############################

EPR_TAU = 0.5
def estimate_epr(preds, tau=EPR_TAU, k_max=1000):
    """ Compute the Effective Progress Resolution,
    formula:
        EPR = -log2( min {delta | delta * |unique( Quantized_delta (preds) )| >= tau} ), 
            where delta in {1 / k | k=1,2,...,k_max}
    
    k_max defines the finest resolution we consider (e.g., k_max=1000 means up to 0.1% resolution)
    """
    def f(preds, delta): # f(delta) = delta * |unique( Quantized_delta (preds) )|
        if delta <= 0:
            return 0
        norm_preds = np.clip(preds, 0, 100) / 100
        quantized = np.floor(norm_preds / delta)
        unique_bins = np.unique(quantized)
        return delta * len(unique_bins)
    
    for k in range(k_max, 0, -1):
        delta = 1.0 / k
        if f(preds, delta) >= tau:
            return float(-np.log2(delta))
    return None # should not reach here as delta=1 always satisfies f(preds,1)>=tau


def compute_epr(pred_txts, episodes, tau=EPR_TAU, k_max=1000):
    """ Group predictions by episode and compute the Effective Progress Resolution,
    formula:
        EPR = -log2( min {delta | delta * |unique( Quantized_delta (preds) )| >= tau} ), 
            where delta in {1 / k | k=1,2,...,k_max}

    k_max defines the finest resolution we consider (e.g., k_max=1000 means up to 0.1% resolution)
    Average the EPR across episodes to get the final score. """
    epi_data = {}

    for p, e in zip(pred_txts, episodes):
        pv = extract_progress_value(p)

        if pv is None:
            continue

        pv = max(0, min(100, pv))
        epi_data.setdefault(e, []).append(float(pv))

    scores = []

    for preds in epi_data.values():
        if len(preds) < 2:
            continue
        preds = np.array(preds, dtype=float)
        epr = estimate_epr(preds, tau, k_max)
        scores.append(float(epr))

    return float(np.mean(scores)) if scores else 0.0


############################
# Matthew's Correlation Coefficient for Success Detection by Progress Thresholding
############################

MCC_THRESHOLD = 95
def compute_mcc_from_progress(pred_txts,gt_txts,thres=MCC_THRESHOLD):
    TP=TN=FP=FN=0

    for p,g in zip(pred_txts,gt_txts):
        pv=extract_progress_value(p)
        gv=extract_progress_value(g)

        if pv is None or gv is None:
            continue

        pred_done = pv>=thres
        gt_done   = gv>=thres

        if pred_done and gt_done: TP+=1
        elif not pred_done and not gt_done: TN+=1
        elif pred_done and not gt_done: FP+=1
        elif not pred_done and gt_done: FN+=1

    denom=np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)+1e-8)

    if denom==0:
        return 0.5  # neutral value after normalization

    mcc=(TP*TN-FP*FN)/denom
    return mcc  # range: [-1,1]


############################
# Kendall Tau
############################

def kendall_tau_a(x, y):
    C = D = 0
    n = len(x)

    for i, j in itertools.combinations(range(n), 2):
        dx = np.sign(x[i] - x[j])
        dy = np.sign(y[i] - y[j])

        if dx == 0 or dy == 0:
            continue  # τ-a ignores ties

        if dx == dy:
            C += 1
        else:
            D += 1

    denom = n * (n - 1) / 2
    if denom == 0:
        return np.nan

    return (C - D) / denom


def compute_kt(pred_txts, gt_txts, episodes):
    epi_data = {}

    for p, g, e in zip(pred_txts, gt_txts, episodes):
        pv = extract_progress_value(p)
        gv = extract_progress_value(g)
        if pv is None or gv is None:
            continue
        epi_data.setdefault(e, []).append((gv, pv))  # (GT, Pred)

    scores = []
    for seq in epi_data.values():
        if len(seq) < 2:
            continue
        seq_sorted = sorted(seq, key=lambda x: x[0])
        y_true = np.array([x[0] for x in seq_sorted], dtype=float)
        y_pred = np.array([x[1] for x in seq_sorted], dtype=float)

        tau = kendall_tau_a(y_true, y_pred)
        if tau is None or np.isnan(tau):
            continue
        scores.append(float(tau))

    return float(np.mean(scores)) if scores else 0.0


############################
# Value-Order Correlation
############################

def compute_voc(pred_txts, gt_txts, episodes):
    epi_data = {}

    for p, g, e in zip(pred_txts, gt_txts, episodes):
        pv = extract_progress_value(p)
        gv = extract_progress_value(g)
        if pv is None or gv is None:
            continue
        epi_data.setdefault(e, []).append((gv, pv))  # (GT, Pred)
    scores = []

    for seq in epi_data.values():
        if len(seq) < 2:
            continue

        seq_sorted = sorted(seq, key=lambda x: x[0])
        pred_values = np.array([x[1] for x in seq_sorted], dtype=float)

        T = len(pred_values)
        time_indices = np.arange(T)

        try:
            corr, _ = spearmanr(time_indices, pred_values)
            if corr is None or np.isnan(corr):
                corr = 0.0
        except Exception:
            corr = 0.0
        if not (-1 <= corr <= 1):
            corr = 0.0
        scores.append(float(corr))

    return float(np.mean(scores)) if scores else 0.0


############################
# Trainer metric
############################
def auto_detect_task_type(gt_txt):
    if "<progress>" in gt_txt and "</progress>" in gt_txt:
        return "progress"
    try:
        gt_json=json_repair.loads(gt_txt)
        if isinstance(gt_json,list) and all(isinstance(x,dict) for x in gt_json):
            return "segmentation"
    except:
        pass
    return "planning"

def _update_average(metrics):
    metric_values = [value for key, value in metrics.items() if key != "average"]
    metrics["average"] = float(np.mean(metric_values)) if metric_values else 0.0
    return metrics

def compute_accuracy(pred_txts,gt_txts,return_average=False,include_deprecated=False):
    seg_tp = 0
    seg_fp = 0
    seg_fn = 0
    seg_abs_error_sum = 0.0
    seg_matched_count = 0
    plan=[]
    for p, g in zip(pred_txts, gt_txts):
        task_type=auto_detect_task_type(g)
        if task_type=="segmentation":
            seg_result = eval_action_segmentation(p, g)
            seg_tp += seg_result["tp"]
            seg_fp += seg_result["fp"]
            seg_fn += seg_result["fn"]
            seg_abs_error_sum += seg_result["matched_abs_error_sum"]
            seg_matched_count += seg_result["matched_count"]
        elif include_deprecated and task_type=="progress":
            # also consider its reasoning part as a planning case for SA metric
            p_plan = "\n".join(p.strip().splitlines()[:-1])
            g_plan = "\n".join(g.strip().splitlines()[:-1])
            plan.append(eval_action_description_or_prediction(p_plan, g_plan))
        elif include_deprecated:
            plan.append(eval_action_description_or_prediction(p,g))
    result={}
    seg_denom = 2 * seg_tp + seg_fp + seg_fn
    if seg_denom > 0:
        result[f"BF1@{int(SEGMENTATION_BOUNDARY_SIGMA)}"] = float(2 * seg_tp / seg_denom)
        result["mMAE"] = float(seg_abs_error_sum / seg_matched_count) if seg_matched_count > 0 else 0.0
    if include_deprecated and plan:
        result["SA"] = np.mean(plan)
    result[f"MCC@{int(MCC_THRESHOLD)}"] = compute_mcc_from_progress(pred_txts,gt_txts)
    if return_average:
        result = _update_average(result)
    return result


############################
# Eval metric with episode grouping
############################

def compute_accuracy_with_episode(pred_txts,gt_txts,episodes,return_average=False,include_deprecated=False):
    metrics = compute_accuracy(pred_txts,gt_txts,include_deprecated=include_deprecated)
    
    # based on episode grouping, compute additional metrics for progress estimation
    progress_preds = []
    progress_gts = []
    progress_eps = []
    for p, g, e in zip(pred_txts, gt_txts, episodes):
        task_type=auto_detect_task_type(g)
        if task_type=="progress":
            progress_preds.append(p)
            progress_gts.append(g)
            progress_eps.append(e)
    
    metrics["CCC"] = compute_ccc(progress_preds, progress_gts, progress_eps)
    metrics[f"EPR@{int(EPR_TAU*100)}"] = compute_epr(progress_preds, progress_eps)
    metrics["VOC"] = compute_voc(progress_preds, progress_gts, progress_eps)
    metrics["KT"] = compute_kt(progress_preds, progress_gts, progress_eps)
    
    if return_average:
        metrics = _update_average(metrics)
    return metrics


############################
# Trainer wrapper
############################

def compute_accuracy_wrapper(eval_pred, tokenizer):
    predictions,labels=eval_pred
    if isinstance(predictions,tuple):
        predictions=predictions[0]
    predictions=np.where(predictions!=-100,predictions,tokenizer.pad_token_id)
    labels=np.where(labels!=-100,labels,tokenizer.pad_token_id)
    decoded_preds=tokenizer.batch_decode(predictions,skip_special_tokens=False)
    decoded_labels=tokenizer.batch_decode(labels,skip_special_tokens=False)
    pred_txts=[p.strip() for p in decoded_preds]
    gt_txts=[l.strip() for l in decoded_labels]
    return compute_accuracy(pred_txts,gt_txts)
