import numba
import numpy as np
import pandas as pd


@numba.njit("f4[:,:](f4[:,:], f4[:,:])", nogil=True, fastmath=True, cache=True)
def dot_product(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute dot product similarity matrix between two embedding matrices."""
    assert X.ndim == 2 and Y.ndim == 2
    assert X.shape[1] == Y.shape[1]

    xlen: int = X.shape[0]
    ylen: int = Y.shape[0]
    embdim: int = X.shape[1]

    emb1_normed: np.ndarray = np.ones((xlen, embdim), dtype=np.float32)
    emb2_normed: np.ndarray = np.ones((ylen, embdim), dtype=np.float32)
    density: np.ndarray = np.empty((xlen, ylen), dtype=np.float32)
    # numba does not support sum() args other then first
    emb1_normed = X / 1
    emb2_normed = Y / 1
    density = emb1_normed @ emb2_normed.T
    return density


@numba.njit("f4[:,:](f4[:,:], f4)", nogil=True, fastmath=True, cache=True)
def fill_matrix_local(a: np.ndarray, gap_extension: float):
    """Fill score matrix for local alignment (Smith-Waterman style)."""
    nrows: int = a.shape[0] + 1
    ncols: int = a.shape[1] + 1
    H: np.ndarray = np.zeros((nrows, ncols), dtype=np.float32)
    h_tmp: np.ndarray = np.zeros(4, dtype=np.float32)
    for i in range(1, nrows):
        for j in range(1, ncols):
            h_tmp[0] = H[i - 1, j - 1] + a[i - 1, j - 1]
            h_tmp[1] = H[i - 1, j] - gap_extension
            h_tmp[2] = H[i, j - 1] - gap_extension
            H[i, j] = np.max(h_tmp)
    return H


@numba.njit("f4[:,:](f4[:,:], f4)", nogil=True, fastmath=True, cache=True)
def fill_matrix_global(a: np.ndarray, gap_extension: float):
    """Fill score matrix for global alignment (Needleman-Wunsch style)."""
    nrows: int = a.shape[0] + 1
    ncols: int = a.shape[1] + 1
    H: np.ndarray = np.zeros((nrows, ncols), dtype=np.float32)
    h_tmp: np.ndarray = np.zeros(3, dtype=np.float32)
    for i in range(0, nrows):
        for j in range(0, ncols):
            if (i == 0) and (j == 0):
                H[i, j] = 0
            elif (i == 0) or (j == 0):
                H[i, j] = -(i + j - 1) * gap_extension
            else:
                h_tmp[0] = H[i - 1, j - 1] + a[i - 1, j - 1]
                h_tmp[1] = H[i - 1, j] - gap_extension
                h_tmp[2] = H[i, j - 1] - gap_extension
                H[i, j] = np.max(h_tmp)
    return H


def fill_score_matrix(sub_matrix: np.ndarray, gap_extension: int | float = 0.0, mode: str = "local") -> np.ndarray:
    """
    Use substitution matrix to create score matrix.
    Set mode = local for Smith-Waterman like procedure (many local alignments)
    and mode = global for Needleman-Wunsch like procedure (one global alignment)
    """
    assert gap_extension >= 0, "gap extension must be positive"
    assert isinstance(mode, str)
    assert mode in {"global", "local"}
    assert isinstance(gap_extension, (int, float))
    assert isinstance(sub_matrix, np.ndarray), "substitution matrix must be numpy array"
    # func fill_matrix require np.float32 array as input
    if not np.issubdtype(sub_matrix.dtype, np.float32):
        sub_matrix = sub_matrix.astype(np.float32)
    if mode == "local":
        score_matrix = fill_matrix_local(sub_matrix, gap_extension=gap_extension)
    elif mode == "global":
        score_matrix = fill_matrix_global(sub_matrix, gap_extension=gap_extension)
    return score_matrix


@numba.njit("types.tuple((f4, i4))(f4, f4, f4)", cache=True)
def max_from_3(x: float, y: float, z: float) -> tuple[float, int]:
    """Return value and index of biggest values."""
    # 2 idx should be diagonal
    if z >= y and z >= x:
        return z, 2
    if x > y and x > z:
        return x, 0
    else:
        return y, 1


@numba.njit("i4[:,:](f4[:,:], types.tuple((i4, i4)), types.unicode_type)", cache=True)
def traceback_from_point_opt2(score_matrix: np.ndarray, max_indice: tuple[int, int], mode: str) -> np.ndarray:
    """Traceback algorithm to find optimal alignment path."""
    assert mode in {"local", "global"}
    y, x = max_indice
    score: float = score_matrix[y, x]

    path = []
    # Fix: Use proper termination condition like original PLMAlign
    while (y > 1) or (x > 1):
        path.append((y, x))

        if mode == "local" and score <= 0:
            break

        # Get scores for three possible moves
        if y > 0 and x > 0:
            diag_score = score_matrix[y - 1, x - 1]
        else:
            diag_score = -np.inf

        if y > 0:
            up_score = score_matrix[y - 1, x]
        else:
            up_score = -np.inf

        if x > 0:
            left_score = score_matrix[y, x - 1]
        else:
            left_score = -np.inf

        # Choose the direction with highest score
        max_score, direction = max_from_3(up_score, left_score, diag_score)

        if direction == 0:  # up
            y -= 1
        elif direction == 1:  # left
            x -= 1
        else:  # diagonal
            y -= 1
            x -= 1

        score = max_score

    # Add final position if we reached boundary
    if y <= 1 or x <= 1:
        path.append((y, x))

    # Convert to numpy array and reverse
    path_array = np.array(path, dtype=np.int32)
    return path_array[::-1]


def plmalign_gather_all_paths(array: np.ndarray, norm: bool = True, mode: str = "local", gap_extension: float = 1.0, with_scores: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Calculate scoring matrix from input substitution matrix and find optimal path.
    """
    assert isinstance(mode, str)
    assert mode in {"global", "local"}
    assert isinstance(gap_extension, (int, float))

    if not isinstance(array, np.ndarray):
        array = array.numpy().astype(np.float32)
    if not isinstance(norm, (str, bool)):
        raise ValueError(f"norm_rows arg should be bool type, but given: {norm}")
    # standardize embedding
    if isinstance(norm, bool):
        if norm:
            arraynorm = (array - array.mean()) / (array.std() + 1e-3)
        else:
            arraynorm = array.copy()
    score_matrix = fill_score_matrix(arraynorm, gap_extension=gap_extension, mode=mode)
    # get all edge indices for left and bottom
    # score_matrix shape array.shape + 1
    # local alignment mode
    if mode == "local":
        indice = np.unravel_index(np.argmax(score_matrix, axis=None), score_matrix.shape)
    # global alignment mode
    elif mode == "global":
        indice = (score_matrix.shape[0] - 1, score_matrix.shape[1] - 1)
    path = traceback_from_point_opt2(score_matrix, indice, mode=mode)
    if with_scores:
        return (path, score_matrix)
    else:
        return path


def plmalign_search_paths(submatrix: np.ndarray, path: np.ndarray, mode: str = "local", as_df: bool = False) -> dict[str, dict] | pd.DataFrame:
    """
    Iterate over path and search for routes matching alignment criteria.
    """
    assert isinstance(submatrix, np.ndarray)
    assert isinstance(mode, str)
    assert mode in {"local", "global"}
    assert isinstance(as_df, bool)

    if not np.issubdtype(submatrix.dtype, np.float32):
        submatrix = submatrix.astype(np.float32)
    spans_locations = {}
    # Adjust path indices (subtract 1 to convert from score matrix coords to sequence coords)
    path = path - 1
    # Filter out negative indices that may have been introduced
    valid_mask = (path >= 0).all(axis=1)
    path = path[valid_mask]

    if len(path) == 0:
        return pd.DataFrame() if as_df else {}

    y, x = path[::-1, 0].ravel(), path[::-1, 1].ravel()
    spans = [(0, len(path))]
    ipath = 0
    if any(spans):
        for idx, (start, stop) in enumerate(spans):
            alnlen = stop - start
            y1, x1 = y[start:stop], x[start:stop]
            arr_values = submatrix[y1, x1]
            arr_indices = np.stack([y1, x1], axis=1)
            keyid = f"{ipath}_{idx}"
            spans_locations[keyid] = {"pathid": 0, "spanid": idx, "span_start": start, "span_end": stop, "indices": arr_indices, "score": arr_values.mean(), "len": alnlen, "mode": mode}
    if as_df:
        return pd.DataFrame(spans_locations.values())
    else:
        return spans_locations


def draw_alignment(indices: np.ndarray, seq1: str, seq2: str, output: str = "str") -> str | dict[str, str]:
    """Draw alignment string from alignment indices."""
    if len(indices) == 0:
        return ""

    # Extract aligned positions
    seq1_positions = indices[:, 0]
    seq2_positions = indices[:, 1]

    # Build alignment strings
    aligned_seq1 = ""
    aligned_seq2 = ""
    match_string = ""

    for i in range(len(indices)):
        pos1, pos2 = seq1_positions[i], seq2_positions[i]
        if 0 <= pos1 < len(seq1) and 0 <= pos2 < len(seq2):
            char1 = seq1[pos1]
            char2 = seq2[pos2]
            aligned_seq1 += char1
            aligned_seq2 += char2
            match_string += "|" if char1 == char2 else " "
        else:
            # Handle gaps (though PLMAlign typically doesn't have explicit gaps)
            aligned_seq1 += "-"
            aligned_seq2 += "-"
            match_string += " "

    if output == "str":
        return f"{aligned_seq1}\n{match_string}\n{aligned_seq2}"
    else:
        return {"seq1_aligned": aligned_seq1, "seq2_aligned": aligned_seq2, "match_string": match_string}
