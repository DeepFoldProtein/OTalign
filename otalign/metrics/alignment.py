from dataclasses import dataclass

import numpy as np


Pair = tuple[int, int]


### set metrics


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


def alignment_scores(pred_pairs: set[Pair], ref_pairs: set[Pair], tolerance: int = 0) -> AlignmentScores:
    pred_size = len(pred_pairs)
    ref_size = len(ref_pairs)

    if tolerance > 0 and pred_size > 0 and ref_size > 0:
        # Greedy matching for tolerant evaluation.
        # To ensure deterministic results, we sort the reference pairs.
        sorted_ref_pairs = sorted(ref_pairs)
        unmatched_preds = set(pred_pairs)

        tp = 0
        for r_x, r_y in sorted_ref_pairs:
            # Find a matching prediction within the tolerance window.
            match_found = None
            # To make matching deterministic, we should also sort the potential matches.
            # However, for a greedy algorithm, iterating over the set and taking the first
            # found match is a valid (though one of many possible) greedy strategies.
            # Sorting preds for each ref would be O(P log P) for each R, which is slow.
            # Let's stick to the simple iteration.
            for p_x, p_y in unmatched_preds:
                if max(abs(p_x - r_x), abs(p_y - r_y)) <= tolerance:
                    match_found = (p_x, p_y)
                    break  # Take the first match found

            if match_found:
                tp += 1
                unmatched_preds.remove(match_found)

        fp = pred_size - tp
        fn = ref_size - tp
    else:
        # Strict matching (tolerance == 0 or no pairs to match)
        tp = len(pred_pairs & ref_pairs)
        fp = len(pred_pairs - ref_pairs)
        fn = len(ref_pairs - pred_pairs)

    if pred_size == 0:
        precision = 1.0 if ref_size == 0 else float("nan")
    else:
        precision = tp / pred_size

    if ref_size == 0:
        recall = 1.0 if pred_size == 0 else float("nan")
    else:
        recall = tp / ref_size

    # Handle cases where precision or recall is NaN
    pr_sum = precision + recall
    if pr_sum != pr_sum or pr_sum == 0:  # check for nan or zero
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / pr_sum

    # For tolerant evaluation, Jaccard is tp / (pred_size + ref_size - tp)
    union = pred_size + ref_size - tp
    if union == 0:
        jaccard = 1.0
    else:
        jaccard = tp / union

    return AlignmentScores(precision=precision, recall=recall, f1=f1, jaccard=jaccard, tp=tp, fp=fp, fn=fn, pred_size=pred_size, ref_size=ref_size)


def build_ref_matrix(ref: set[Pair], m: int, n: int) -> np.ndarray:
    T = np.zeros((m, n))
    for i, j in ref:
        T[i, j] = 1
    return T


def soft_alignment_scores(prob_match: np.ndarray, ref_matrix: np.ndarray, eps: float = 1e-5) -> dict[str, float]:
    tp_soft = np.sum(prob_match * ref_matrix)
    ENM = np.sum(prob_match)
    ref_size = ref_matrix.sum()

    precision = tp_soft / (ENM + eps)
    recall = tp_soft / (ref_size + eps)
    pr_sum = precision + recall
    if pr_sum != pr_sum or pr_sum == 0:  # check for nan or zero
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / pr_sum

    return {"soft_precision": precision, "soft_recall": recall, "soft_f1": f1}
