from dataclasses import dataclass

import torch
import torch.nn.functional as F


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


def alignment_scores(pred_pairs: set[Pair], ref_pairs: set[Pair]) -> AlignmentScores:
    tp = len(pred_pairs & ref_pairs)
    fp = len(pred_pairs - ref_pairs)
    fn = len(ref_pairs - pred_pairs)

    pred_size = len(pred_pairs)
    ref_size = len(ref_pairs)

    if pred_size == 0:
        precision = 1.0 if ref_size == 0 else float("nan")
    else:
        precision = tp / pred_size

    if ref_size == 0:
        recall = 1.0 if pred_size == 0 else float("nan")
    else:
        recall = tp / ref_size

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    union = len(pred_pairs | ref_pairs)
    if union == 0:
        jaccard = 1.0
    else:
        jaccard = tp / union

    return AlignmentScores(precision=precision, recall=recall, f1=f1, jaccard=jaccard, tp=tp, fp=fp, fn=fn, pred_size=pred_size, ref_size=ref_size)


### plan metrics


def in_band_mass(pred_plan: torch.Tensor, true_plan: torch.Tensor, band_width: int) -> float:
    """
    정답 정렬 주변의 밴드 내에 포함된 확률 질량을 계산합니다. (Precision-like)

    Args:
        pred_plan (torch.Tensor): 정규화된(총합=1) 예측 확률 행렬.
        true_plan (torch.Tensor): 정답 정렬 행렬 (0 또는 1).
        band_width (int): 밴드의 절반 폭 (w).

    Returns:
        float: 밴드 내의 총 확률 질량 (0과 1 사이의 값).
    """
    if true_plan.sum() == 0:
        return 0.0

    # 2D Max Pooling을 사용하여 밴드 마스크를 효율적으로 생성합니다.
    kernel_size = 2 * band_width + 1
    padding = band_width
    # true_plan을 4D 텐서로 변환: (N, C, H, W)
    true_plan_4d = true_plan.unsqueeze(0).unsqueeze(0)

    # max_pool2d를 적용하여 각 픽셀 주변의 최대값을 찾습니다.
    # 정답 위치(1)가 있으면 주변 영역이 1로 채워진 밴드 마스크가 생성됩니다.
    band_mask = F.max_pool2d(true_plan_4d, kernel_size=kernel_size, stride=1, padding=padding)
    band_mask = band_mask.squeeze(0).squeeze(0) > 0.5

    # 예측 확률 행렬과 밴드 마스크를 곱하여 밴드 내의 확률만 남깁니다.
    mass = torch.sum(pred_plan * band_mask.float())

    return mass.item()


def recall_in_band(pred_plan: torch.Tensor, true_plan: torch.Tensor, band_width: int) -> float:
    """
    Hard threshold 없이, 각 정답 위치 주변에 포착된 확률 질량의 평균을
    'Soft'한 점수로 계산합니다. (Robust & Soft Recall-like)

    Args:
        pred_plan (torch.Tensor): 정규화된 예측 확률 행렬.
        true_plan (torch.Tensor): 정답 정렬 행렬.
        band_width (int): 국소 밴드의 절반 폭.

    Returns:
        float: Soft Recall 점수.
    """
    num_true = true_plan.sum()
    if num_true == 0:
        return 1.0

    kernel_size = 2 * band_width + 1
    plan_4d = pred_plan.unsqueeze(0).unsqueeze(0)

    sum_pooled = F.avg_pool2d(plan_4d, kernel_size=kernel_size, stride=1, padding=band_width) * (kernel_size * kernel_size)

    true_coords = torch.nonzero(true_plan > 0.5, as_tuple=False)

    captured_masses = sum_pooled[0, 0, true_coords[:, 0], true_coords[:, 1]]

    # [수정된 부분] 각 정답 위치의 점수가 1.0을 넘지 않도록 보정합니다.
    # 중복 계산으로 인해 합계가 1을 약간 넘는 경우를 방지하여 안정성을 높입니다.
    captured_masses.clamp_(max=1.0)

    # Hard threshold 없이, 포착된 확률 질량의 평균을 그대로 사용합니다.
    recall = torch.mean(captured_masses)

    return recall.item()


def vectorized_in_band_mass(pred_plans: torch.Tensor, true_plans: torch.Tensor, band_width: int) -> torch.Tensor:
    """
    Vectorized version of in_band_mass for a batch of plans.
    """
    if true_plans.sum() == 0:
        return torch.zeros(pred_plans.shape[0], device=pred_plans.device)

    kernel_size = 2 * band_width + 1
    padding = band_width

    # Add channel dimension for pooling
    true_plans_4d = true_plans.unsqueeze(1)  # (B, 1, H, W)

    band_mask = F.max_pool2d(true_plans_4d, kernel_size=kernel_size, stride=1, padding=padding)
    band_mask = band_mask.squeeze(1) > 0.5  # (B, H, W)

    mass = torch.sum(pred_plans * band_mask.float(), dim=(1, 2))  # (B,)
    return mass


def vectorized_recall_in_band(pred_plans: torch.Tensor, true_plans: torch.Tensor, band_width: int) -> torch.Tensor:
    """
    Fully vectorized version of recall_in_band.
    """
    num_true = true_plans.sum(dim=(1, 2))
    # Create a mask for samples that have true alignments
    has_true = num_true > 0

    # Initialize recalls with 1.0 for samples with no true alignments
    recalls = torch.ones(pred_plans.shape[0], device=pred_plans.device)

    if not has_true.any():
        return recalls

    # Filter plans that have true alignments
    pred_plans_filt = pred_plans[has_true]
    true_plans_filt = true_plans[has_true]
    num_true_filt = num_true[has_true]

    kernel_size = 2 * band_width + 1
    padding = band_width

    plans_4d = pred_plans_filt.unsqueeze(1)  # (B_filt, 1, H, W)

    sum_pooled = F.avg_pool2d(plans_4d, kernel_size=kernel_size, stride=1, padding=padding) * (kernel_size * kernel_size)
    sum_pooled = sum_pooled.squeeze(1)  # (B_filt, H, W)

    # Clamp to avoid scores > 1.0
    sum_pooled.clamp_(max=1.0)

    # Multiply by the true plan to get captured mass at each true location
    captured_mass_total = torch.sum(sum_pooled * true_plans_filt, dim=(1, 2))

    # Calculate mean recall for each sample
    recall_filt = captured_mass_total / num_true_filt

    # Update the recalls tensor at the correct indices
    recalls[has_true] = recall_filt

    return recalls
