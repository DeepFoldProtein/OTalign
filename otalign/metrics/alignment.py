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
        precision = 1.0 if ref_size == 0 else 0.0
    else:
        precision = tp / pred_size

    if ref_size == 0:
        recall = 1.0 if pred_size == 0 else 0.0
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
    # 정답이 없는 경우, 밴드 내 질량은 0입니다.
    if true_plan.sum() == 0:
        return 0.0

    # 밴드 영역을 표시할 마스크를 생성합니다.
    band_mask = torch.zeros_like(true_plan, dtype=torch.bool)
    true_coords = torch.nonzero(true_plan > 0.5, as_tuple=False)

    # 각 정답 좌표 주변의 (2w+1) x (2w+1) 영역을 마스크에 1로 표시합니다.
    for r, c in true_coords:
        r_min = max(0, r - band_width)
        r_max = min(band_mask.shape[0], r + band_width + 1)
        c_min = max(0, c - band_width)
        c_max = min(band_mask.shape[1], c + band_width + 1)
        band_mask[r_min:r_max, c_min:c_max] = True

    # 예측 확률 행렬과 밴드 마스크를 곱하여 밴드 내의 확률만 남깁니다.
    mass = torch.sum(pred_plan * band_mask.float())

    return mass.item()


def recall_in_band(pred_plan: torch.Tensor, true_plan: torch.Tensor, band_width: int) -> float:
    """
    각 정답 위치 주변의 국소 밴드에 포함된 예측 확률의 평균을 계산합니다. (Recall-like)

    Args:
        pred_plan (torch.Tensor): 정규화된(총합=1) 예측 확률 행렬.
        true_plan (torch.Tensor): 정답 정렬 행렬 (0 또는 1).
        band_width (int): 국소 밴드의 절반 폭 (w).

    Returns:
        float: 정답 위치 당 평균적으로 포착된 확률 질량.
    """
    # 정답이 없는 경우, 모든 정답을 찾았다고 간주하여 1.0을 반환합니다.
    if true_plan.sum() == 0:
        return 1.0

    # 커널(밴드) 크기를 정의합니다.
    kernel_size = 2 * band_width + 1

    # F.avg_pool2d는 4D 텐서(N, C, H, W)를 입력으로 받으므로 차원을 추가합니다.
    # (H, W) -> (1, 1, H, W)
    plan_4d = pred_plan.unsqueeze(0).unsqueeze(0)

    # Average Pooling을 사용하여 각 위치 주변의 '합계'를 효율적으로 계산합니다.
    # padding을 추가하여 행렬 경계에서도 밴드 크기를 동일하게 유지합니다.
    # avg_pool2d의 결과에 커널 넓이를 곱하면 sum_pool2d와 동일한 효과를 냅니다.
    sum_pooled = F.avg_pool2d(plan_4d, kernel_size=kernel_size, stride=1, padding=band_width) * (kernel_size * kernel_size)

    # 정답 좌표를 가져옵니다.
    true_coords = torch.nonzero(true_plan > 0.5, as_tuple=False)

    # pooling된 결과에서 정답 좌표에 해당하는 값들만 추출합니다.
    # sum_pooled는 (1, 1, H, W) 형태이므로, 인덱싱을 맞춥니다.
    captured_masses = sum_pooled[0, 0, true_coords[:, 0], true_coords[:, 1]]

    # 추출된 값들의 평균을 계산하여 최종 점수를 얻습니다.
    recall = torch.mean(captured_masses)

    return recall.item()
