import numpy as np


def aln_to_matches(aln1, aln2):
    if all("-" not in x for x in (aln1, aln2)):
        return []

    matches = []
    i, j = 0, 0
    for a, b in zip(aln1, aln2):
        if a != "-" and b != "-":
            matches.append((i, j))
            i += 1
            j += 1
        elif a != "-":
            i += 1
        elif b != "-":
            j += 1
        else:
            continue

    return matches


def convolve2d(image, kernel, padding_mode="valid", stride=1):
    """
    Performs 2D convolution of an image with a given kernel.
    Optimized for performance using NumPy's vectorized operations.

    Args:
        image (np.ndarray): The input 2D image (grayscale).
        kernel (np.ndarray): The 2D convolution kernel.
        padding_mode (str): The padding mode to use.
                            'valid': No padding. Output size will be smaller.
                            'same': Pads the image so that the output size is the same as the input.
                                    Pads with zeros.
                            'full': Pads the image so that every pixel of the input is covered
                                    by the kernel. Output size will be larger.
                            'reflect', 'symmetric', 'edge', 'wrap': Other NumPy padding modes.
                                    (Note: 'same' and 'full' specifically use 'constant' padding with 0s)
        stride (int): The stride of the convolution. Defaults to 1.

    Returns:
        np.ndarray: The convolved image.

    Raises:
        ValueError: If the padding_mode is not recognized or if stride is less than 1.
    """

    if stride < 1:
        raise ValueError("Stride must be at least 1.")

    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate padding amounts based on mode
    if padding_mode == "valid":
        pad_height_top, pad_height_bottom = 0, 0
        pad_width_left, pad_width_right = 0, 0
    elif padding_mode == "same":
        # Calculate total padding needed
        total_pad_height = kernel_height - 1
        total_pad_width = kernel_width - 1

        # Distribute padding: prefer more padding on the bottom/right if odd
        pad_height_top = total_pad_height // 2
        pad_height_bottom = total_pad_height - pad_height_top
        pad_width_left = total_pad_width // 2
        pad_width_right = total_pad_width - pad_width_left
    elif padding_mode == "full":
        pad_height_top, pad_height_bottom = kernel_height - 1, kernel_height - 1
        pad_width_left, pad_width_right = kernel_width - 1, kernel_width - 1
    else:
        # For other NumPy padding modes, we apply padding to achieve a 'same'-like output size
        # if the user doesn't explicitly specify padding amounts.
        # This is a heuristic to make these modes behave reasonably for convolution.
        total_pad_height = kernel_height - 1
        total_pad_width = kernel_width - 1
        pad_height_top = total_pad_height // 2
        pad_height_bottom = total_pad_height - pad_height_top
        pad_width_left = total_pad_width // 2
        pad_width_right = total_pad_width - pad_width_left

    # Apply padding
    if padding_mode == "valid":
        padded_image = image
    elif padding_mode in ["same", "full"]:
        # For 'same' and 'full', we use constant padding with zeros
        padded_image = np.pad(
            image,
            ((pad_height_top, pad_height_bottom), (pad_width_left, pad_width_right)),
            mode="constant",
            constant_values=0,
        )
    else:
        # For other specified NumPy padding modes
        try:
            padded_image = np.pad(
                image,
                ((pad_height_top, pad_height_bottom), (pad_width_left, pad_width_right)),
                mode=padding_mode,
            )
        except ValueError as e:
            raise ValueError(f"Invalid padding_mode '{padding_mode}' or padding calculation error: {e}")

    padded_image_height, padded_image_width = padded_image.shape

    # Calculate output dimensions
    output_height = (padded_image_height - kernel_height) // stride + 1
    output_width = (padded_image_width - kernel_width) // stride + 1

    # Ensure output dimensions are non-negative
    if output_height <= 0 or output_width <= 0:
        raise ValueError("Output dimensions are non-positive. Check image, kernel, padding, and stride values.")

    # Get strides of the padded image
    image_strides = padded_image.strides

    # Create a view of the padded image that represents all possible kernel-sized windows
    # This is the core of the vectorized approach.
    # The new shape will be (output_height, output_width, kernel_height, kernel_width)
    # The new strides will allow us to slide the window by 'stride' for the first two dimensions
    # and then access elements within the kernel for the last two dimensions.
    view_shape = (output_height, output_width, kernel_height, kernel_width)
    view_strides = (
        image_strides[0] * stride,  # Stride for moving down rows in the output
        image_strides[1] * stride,  # Stride for moving right columns in the output
        image_strides[0],  # Stride for moving down rows within the kernel window
        image_strides[1],  # Stride for moving right columns within the kernel window
    )

    # Use as_strided to create the view without copying data
    sub_matrices = np.lib.stride_tricks.as_strided(padded_image, shape=view_shape, strides=view_strides)

    # Perform element-wise multiplication with the kernel and sum along the kernel dimensions
    # np.einsum is very efficient for this type of operation (sum-product)
    # 'ijkl,kl->ij' means:
    # i,j: output dimensions
    # k,l: kernel dimensions
    # For each (i,j) in the output, sum over k,l of (sub_matrices[i,j,k,l] * kernel[k,l])
    output = np.einsum("ijkl,kl->ij", sub_matrices, kernel)

    return output
