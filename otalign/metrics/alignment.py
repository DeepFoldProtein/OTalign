from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


Pair = Tuple[int, int]


def _to_pair_set(ref_alignment: Iterable[Sequence[int]]) -> Set[Pair]:
    return {(int(i), int(j)) for i, j in ref_alignment}


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def _check_plan(plan: np.ndarray) -> None:
    if plan.ndim != 2:
        raise ValueError("plan must be a 2D array of shape [m, n].")
    if not np.isfinite(plan).all():
        raise ValueError("plan contains non-finite values.")


### Plan to discrete pairs extractors


def predict_pairs_threshold(plan: np.ndarray, threshold: float) -> Set[Pair]:
    """Many-to-many: all entries >= threshold."""
    _check_plan(plan)
    idx = np.argwhere(plan >= threshold)
    return {(int(i), int(j)) for i, j in idx}


def predict_pairs_topk_per_row(plan: np.ndarray, k: int) -> Set[Pair]:
    """At most k pairs per row (many-to-few)."""
    _check_plan(plan)
    m, n = plan.shape
    pairs: Set[Pair] = set()
    k = max(0, int(k))
    if k == 0:
        return pairs
    # argsort descending per row, take top-k
    order = np.argpartition(-plan, kth=min(k - 1, n - 1), axis=1)
    for i in range(m):
        cols = order[i, : min(k, n)]
        # break ties by actual values descending
        cols = cols[np.argsort(-plan[i, cols])]
        for j in cols:
            pairs.add((int(i), int(j)))
    return pairs


def predict_pairs_bipartite_matching(plan: np.ndarray) -> Set[Pair]:
    """One-to-one maximum-weight matching via Hungarian algorithm."""
    _check_plan(plan)
    # Hungarian solves min-cost; convert to cost = -weight (stable with large constant)
    w = plan.astype(np.float64)
    max_w = np.nanmax(w) if w.size else 0.0
    cost = max_w - w  # non-negative costs; same argmin as -w
    row_ind, col_ind = linear_sum_assignment(cost)
    return {(int(i), int(j)) for i, j in zip(row_ind, col_ind)}


### Set metrics


@dataclass(frozen=True)
class AlignmentScores:
    precision: float
    recall: float
    f1: float
    jaccard: float  # = |∩| / |∪|, sometimes called set accuracy
    tp: int
    fp: int
    fn: int
    pred_size: int
    ref_size: int


def alignment_scores(pred_pairs: Set[Pair], ref_pairs: Set[Pair]) -> AlignmentScores:
    inter = pred_pairs & ref_pairs
    tp = len(inter)
    fp = len(pred_pairs - ref_pairs)
    fn = len(ref_pairs - pred_pairs)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
    jaccard = _safe_div(tp, len(pred_pairs | ref_pairs))
    return AlignmentScores(precision=precision, recall=recall, f1=f1, jaccard=jaccard, tp=tp, fp=fp, fn=fn, pred_size=len(pred_pairs), ref_size=len(ref_pairs))


### High-level helpers


def evaluate_with_threshold(plan: np.ndarray, ref_alignment: Iterable[Sequence[int]], threshold: float = 0.01) -> AlignmentScores:
    ref_pairs = _to_pair_set(ref_alignment)
    pred_pairs = predict_pairs_threshold(plan, threshold)
    return alignment_scores(pred_pairs, ref_pairs)


def evaluate_with_topk(plan: np.ndarray, ref_alignment: Iterable[Sequence[int]], k: int = 1) -> AlignmentScores:
    ref_pairs = _to_pair_set(ref_alignment)
    pred_pairs = predict_pairs_topk_per_row(plan, k)
    return alignment_scores(pred_pairs, ref_pairs)


def evaluate_with_matching(plan: np.ndarray, ref_alignment: Iterable[Sequence[int]]) -> AlignmentScores:
    ref_pairs = _to_pair_set(ref_alignment)
    pred_pairs = predict_pairs_bipartite_matching(plan)
    return alignment_scores(pred_pairs, ref_pairs)


### PR curve and best-F1


@dataclass(frozen=True)
class PRPoint:
    threshold: float
    precision: float
    recall: float
    f1: float


def pr_curve_threshold_sweep(plan: np.ndarray, ref_alignment: Iterable[Sequence[int]], thresholds: Optional[Sequence[float]] = None) -> List[PRPoint]:
    """Sweep thresholds (high->low) and compute PR/F1."""
    _check_plan(plan)
    ref_pairs = _to_pair_set(ref_alignment)

    if thresholds is None:
        # Derive unique sorted scores present in plan to get exact breakpoints.
        uniq = np.unique(plan[np.isfinite(plan)])
        thresholds = uniq[::-1].tolist()  # descending

    out: List[PRPoint] = []
    for t in thresholds:  # type: ignore
        pred_pairs = predict_pairs_threshold(plan, float(t))
        s = alignment_scores(pred_pairs, ref_pairs)
        out.append(PRPoint(threshold=float(t), precision=s.precision, recall=s.recall, f1=s.f1))
    return out


def best_f1_by_threshold(plan: np.ndarray, ref_alignment: Iterable[Sequence[int]], thresholds: Optional[Sequence[float]] = None) -> PRPoint:
    curve = pr_curve_threshold_sweep(plan, ref_alignment, thresholds)
    if not curve:
        return PRPoint(threshold=1.0, precision=0.0, recall=0.0, f1=0.0)
    # choose the best F1; break ties by higher recall, then higher threshold
    best = max(curve, key=lambda p: (p.f1, p.recall, p.threshold))
    return best
