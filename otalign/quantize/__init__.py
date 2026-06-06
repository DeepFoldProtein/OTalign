from typing import Any, Dict, cast

import numpy as np
from numpy.typing import DTypeLike


def quantize(plan: np.ndarray, dtype: DTypeLike = np.uint8) -> Dict[str, Any]:
    """
    Quantizes a dense plan.

    The quantization assumes the minimum value of the plan is 0.0.

    Args:
        plan: The dense float32 transport plan.
        dtype: The target integer dtype for quantization.

    Returns:
        A dictionary containing the quantized representation.
    """
    if np.any(plan < 0):
        raise ValueError("Transport plan contains negative values.")

    q_info = np.iinfo(dtype)
    q_min, q_max = q_info.min, q_info.max

    r_max = plan.max()

    # Quantize with r_min = 0
    scale = r_max / (q_max - q_min) if r_max > 1e-9 else 1.0
    zero_point = q_min

    quantized_plan = np.round(plan / scale + zero_point)
    quantized_plan = np.clip(quantized_plan, q_min, q_max).astype(dtype)

    return {
        "data": quantized_plan,
        "scale": scale,
        "zero_point": zero_point,
        # Self-describing record of how the plan was quantized so that a reader
        # can dequantize correctly even if the scheme changes in the future.
        "meta": {
            "version": 1,
            # decoded = (data - zero_point) * scale
            "scheme": "linear_affine",
            "dtype": np.dtype(dtype).name,
            "r_min": 0.0,  # quantization assumes the plan minimum is 0.0
            "r_max": float(r_max),
            "q_min": int(q_min),
            "q_max": int(q_max),
            "clipped": True,
        },
    }


def dequantize(quantized_data: Dict[str, Any]) -> np.ndarray:
    """
    Decodes a plan from the quantized format.
    """
    data = cast(np.ndarray, quantized_data["data"])
    scale = cast(float, quantized_data["scale"])
    zero_point = cast(int, quantized_data["zero_point"])

    # Dequantize
    decoded_plan = (data.astype(np.float32) - zero_point) * scale
    return decoded_plan
