from typing import Any, Dict, Optional, cast

import numpy as np
from numpy.typing import DTypeLike


# Quantization scheme that dequantize() knows how to decode. Bumped only when the
# on-disk format changes in a way that older decoders cannot read.
SCHEME = "linear_affine"
SCHEME_VERSION = 1


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
            "version": SCHEME_VERSION,
            # decoded = (data - zero_point) * scale
            "scheme": SCHEME,
            "dtype": np.dtype(dtype).name,
            "r_min": 0.0,  # quantization assumes the plan minimum is 0.0
            "r_max": float(r_max),
            "q_min": int(q_min),
            "q_max": int(q_max),
            "clipped": True,
        },
    }


def validate_meta(meta: Optional[Dict[str, Any]], data: Optional[np.ndarray] = None) -> None:
    """
    Check that a saved quantization ``meta`` is compatible with ``dequantize()``.

    Raises ``ValueError`` on an incompatible scheme/version, or a dtype that
    disagrees with the quantized array. A ``None`` meta — legacy files written
    before metadata existed — passes silently, since those are decoded by the
    fixed ``dequantize()`` formula.
    """
    if meta is None:
        return

    scheme = meta.get("scheme")
    if scheme != SCHEME:
        raise ValueError(f"Unsupported quantization scheme {scheme!r}; this build only decodes {SCHEME!r}.")

    version = meta.get("version")
    if version != SCHEME_VERSION:
        raise ValueError(f"Unsupported quantization meta version {version!r}; this build only decodes version {SCHEME_VERSION}.")

    if data is not None and "dtype" in meta and np.dtype(meta["dtype"]) != data.dtype:
        raise ValueError(f"Quantized data dtype {data.dtype} does not match meta dtype {meta['dtype']!r}.")


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
