import torch
import torch.nn.functional as F


def pad_right(
    x: torch.Tensor,
    target_size: int,
    axis: int = -1,
    pad_value: float = 0.0,
):
    """Right-pad a tensor along a single axis up to target_size, filling with pad_value,
    and return an integer mask (1 = real, 0 = pad) of the same rank.

    Args:
        x (torch.Tensor): input tensor of any shape.
        target_size (int): desired size along `axis` after padding.
        axis (int): which axis to pad (can be negative). Defaults to -1 (last axis).
        pad_value (float): value to use for padding. Defaults to 0.0.

    Returns:
        padded (torch.Tensor): same dtype as x, with shape[axis] == target_size.
        mask (torch.LongTensor): same shape as padded, 1 for original elements, 0 for padding.

    """
    ndim = x.dim()
    # normalize axis
    if axis < 0:
        axis = ndim + axis
    orig_size = x.size(axis)

    # if already long enough, just truncate or return as-is
    if orig_size >= target_size:
        # truncate if too long
        slices = [slice(None)] * ndim
        slices[axis] = slice(0, target_size)
        padded = x[tuple(slices)]
        mask = x.new_ones(padded.shape, dtype=torch.bool)
        return padded, mask

    # compute pad amounts: F.pad wants a list [pad_last_dim_left, pad_last_dim_right, ..., pad_first_dim_right]
    pad_amt = target_size - orig_size
    pad_list = [0] * (2 * ndim)
    # for dimension `axis`, we want to pad on the *right* side:
    #   pad_list[2*(ndim-1-axis) + 1] = pad_amt
    pad_list[2 * (ndim - 1 - axis) + 1] = pad_amt

    padded = F.pad(x, pad=pad_list, value=pad_value)

    # build mask
    mask_shape = list(x.shape)
    mask_shape[axis] = target_size
    mask = x.new_zeros(mask_shape, dtype=torch.bool)
    # fill ones for the original region
    ones_slices = [slice(None)] * ndim
    ones_slices[axis] = slice(0, orig_size)
    mask[tuple(ones_slices)] = 1

    return padded, mask
