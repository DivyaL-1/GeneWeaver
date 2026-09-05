import numpy as np

if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack


import os
import math
import time
import subprocess
import numpy as np
from numba import cuda, int8, int32
from dask.distributed import Client, LocalCluster, as_completed as dask_as_completed


BASE_TO_CODE = {"A": 0, "C": 1, "G": 2, "T": 3}
UNKNOWN_CODE = 4


_CODE_LUT = np.full(256, UNKNOWN_CODE, dtype=np.int8)
for _ch, _code in BASE_TO_CODE.items():
    _CODE_LUT[ord(_ch)] = _code
    _CODE_LUT[ord(_ch.lower())] = _code


def encode_sequence(seq):
    if isinstance(seq, np.ndarray):
        return seq.astype(np.int8, copy=False)
    raw = np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)
    return _CODE_LUT[raw]


@cuda.jit
def nw_diagonal_kernel(seq1_d, seq2_d, dp_d, n, m, d, i_lo, i_hi, match, mismatch, gap):
    tid = cuda.grid(1)
    i = i_lo + tid
    if i > i_hi:
        return  # thread has no cell on this diagonal

    j = d - i
    if i == 0 or j == 0:
        return

    row_stride = m + 1
    score = match if seq1_d[i - 1] == seq2_d[j - 1] else mismatch

    diag_val = dp_d[(i - 1) * row_stride + (j - 1)] + score
    up_val = dp_d[(i - 1) * row_stride + j] + gap
    left_val = dp_d[i * row_stride + (j - 1)] + gap

    best = diag_val
    if up_val > best:
        best = up_val
    if left_val > best:
        best = left_val

    dp_d[i * row_stride + j] = best


def needleman_wunsch_cuda(
    seq1_str, seq2_str,
    match=1, mismatch=-1, gap=-2,
    threads_per_block=32,
    progress_callback=None,
):
    n, m = sequence_length(seq1_str), sequence_length(seq2_str)
    row_stride = m + 1

    seq1_codes = encode_sequence(seq1_str)
    seq2_codes = encode_sequence(seq2_str)

    dp_init = np.zeros((n + 1) * (m + 1), dtype=np.int32)
    dp_init[0::row_stride] = np.arange(n + 1, dtype=np.int64) * gap
    dp_init[:row_stride] = np.arange(row_stride, dtype=np.int64) * gap

    timings = {}

    t0 = time.perf_counter()
    seq1_d = cuda.to_device(seq1_codes)
    seq2_d = cuda.to_device(seq2_codes)
    dp_d = cuda.to_device(dp_init)
    cuda.synchronize()
    timings["host_to_device_s"] = time.perf_counter() - t0


    total_diagonals = n + m - 1  
    update_every = max(1, total_diagonals // 100)  

    t1 = time.perf_counter()
    for step, d in enumerate(range(2, n + m + 1), start=1):
        i_lo = max(1, d - m)
        i_hi = min(n, d - 1)
        num_cells = i_hi - i_lo + 1
        if num_cells <= 0:
            continue

        blocks = math.ceil(num_cells / threads_per_block)
        nw_diagonal_kernel[blocks, threads_per_block](
            seq1_d, seq2_d, dp_d, n, m, d, i_lo, i_hi, match, mismatch, gap
        )

        if progress_callback is not None and (step % update_every == 0 or step == total_diagonals):
            cuda.synchronize()  # only sync when we actually need to report progress
            progress_callback(step, total_diagonals)

    cuda.synchronize()
    timings["kernel_exec_s"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    dp_result = dp_d.copy_to_host()
    timings["device_to_host_s"] = time.perf_counter() - t2

    score = int(dp_result[n * row_stride + m])
    timings["total_s"] = sum(timings.values())

    return {"score": score, "n": n, "m": m, "timings": timings}


TILE_DIM = 32  # also the block size; one warp per block, shared tile is 33*33*4 B

TILE_DIM_P1 = 33
assert TILE_DIM_P1 == TILE_DIM + 1, "TILE_DIM_P1 must stay TILE_DIM + 1"


@cuda.jit
def nw_tiled_shared_kernel(
    seq1_d, seq2_d, dp_d, n, m, tile_diag, ti_lo, ti_hi, match, mismatch, gap
):
    s_dp = cuda.shared.array(shape=(TILE_DIM_P1, TILE_DIM_P1), dtype=int32)
    s_s1 = cuda.shared.array(shape=TILE_DIM, dtype=int8)
    s_s2 = cuda.shared.array(shape=TILE_DIM, dtype=int8)

    ti = ti_lo + cuda.blockIdx.x
    if ti > ti_hi:
        return
    tj = tile_diag - ti

    row_stride = m + 1
    i0 = ti * TILE_DIM + 1   # first DP row owned by this tile (1-based)
    j0 = tj * TILE_DIM + 1   # first DP column owned by this tile
    rows = min(TILE_DIM, n - i0 + 1)
    cols = min(TILE_DIM, m - j0 + 1)
    if rows <= 0 or cols <= 0:
        return  # ragged edge tile with nothing in it (uniform across the block)

    tx = cuda.threadIdx.x

    if tx == 0:
        s_dp[0, 0] = dp_d[(i0 - 1) * row_stride + (j0 - 1)]
    if tx < cols:
        s_dp[0, tx + 1] = dp_d[(i0 - 1) * row_stride + (j0 + tx)]
        s_s2[tx] = seq2_d[j0 - 1 + tx]
    if tx < rows:
        s_dp[tx + 1, 0] = dp_d[(i0 + tx) * row_stride + (j0 - 1)]
        s_s1[tx] = seq1_d[i0 - 1 + tx]

    cuda.syncthreads()

    for k in range(rows + cols - 1):
        c = k - tx
        if tx < rows and c >= 0 and c < cols:
            score = match if s_s1[tx] == s_s2[c] else mismatch
            diag_val = s_dp[tx, c] + score
            up_val = s_dp[tx, c + 1] + gap
            left_val = s_dp[tx + 1, c] + gap

            best = diag_val
            if up_val > best:
                best = up_val
            if left_val > best:
                best = left_val

            s_dp[tx + 1, c + 1] = best
        cuda.syncthreads()  # uniform across the block: rows/cols are block-wide

    if tx < cols:
        dp_d[(i0 + rows - 1) * row_stride + (j0 + tx)] = s_dp[rows, tx + 1]
    if tx < rows:
        dp_d[(i0 + tx) * row_stride + (j0 + cols - 1)] = s_dp[tx + 1, cols]


def needleman_wunsch_cuda_shared(
    seq1_str, seq2_str,
    match=1, mismatch=-1, gap=-2,
    progress_callback=None,
):
    n, m = sequence_length(seq1_str), sequence_length(seq2_str)
    row_stride = m + 1

    seq1_codes = encode_sequence(seq1_str)
    seq2_codes = encode_sequence(seq2_str)

    dp_init = np.zeros((n + 1) * (m + 1), dtype=np.int32)
    dp_init[0::row_stride] = np.arange(n + 1, dtype=np.int64) * gap
    dp_init[:row_stride] = np.arange(row_stride, dtype=np.int64) * gap

    timings = {}

    t0 = time.perf_counter()
    seq1_d = cuda.to_device(seq1_codes)
    seq2_d = cuda.to_device(seq2_codes)
    dp_d = cuda.to_device(dp_init)
    cuda.synchronize()
    timings["host_to_device_s"] = time.perf_counter() - t0

    n_ti = math.ceil(n / TILE_DIM)
    n_tj = math.ceil(m / TILE_DIM)
    total_tile_diagonals = n_ti + n_tj - 1
    update_every = max(1, total_tile_diagonals // 100)

    t1 = time.perf_counter()
    for step, tile_diag in enumerate(range(total_tile_diagonals), start=1):
        ti_lo = max(0, tile_diag - (n_tj - 1))
        ti_hi = min(n_ti - 1, tile_diag)
        blocks = ti_hi - ti_lo + 1
        if blocks <= 0:
            continue

        nw_tiled_shared_kernel[blocks, TILE_DIM](
            seq1_d, seq2_d, dp_d, n, m, tile_diag, ti_lo, ti_hi, match, mismatch, gap
        )

        if progress_callback is not None and (
            step % update_every == 0 or step == total_tile_diagonals
        ):
            cuda.synchronize()
            progress_callback(step, total_tile_diagonals)

    cuda.synchronize()
    timings["kernel_exec_s"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    idx = n * row_stride + m
    try:
        score = int(dp_d[idx:idx + 1].copy_to_host()[0])
    except Exception:
        score = int(dp_d.copy_to_host()[idx])
    timings["device_to_host_s"] = time.perf_counter() - t2

    timings["total_s"] = sum(timings.values())

    return {"score": score, "n": n, "m": m, "timings": timings, "kernel": "tiled_shared"}


DEFAULT_KERNEL = "warp"

SHARED_KERNEL_FALLBACK = True   # fall back to the diagonal kernel if a tiled one won't compile
_SHARED_KERNEL_ERROR = None     # set once, so the fallback is reported rather than silent


def shared_kernel_error():
    return _SHARED_KERNEL_ERROR


class DeviceMemoryTooSmall(MemoryError):
    pass

def _guard_full_matrix(n, m, kernel):
    needed = estimate_device_bytes(n, m, kernel)
    fits, _needed, free = check_device_capacity(n, m, kernel)
    if not fits:
        free_txt = f"{free / 1e9:.2f} GB free" if free is not None else "unknown free VRAM"
        raise DeviceMemoryTooSmall(
            f"kernel '{kernel}' needs a {n + 1} x {m + 1} int32 DP matrix "
            f"= {needed / 1e9:,.2f} GB of VRAM ({free_txt}). "
            f"Use kernel='{DEFAULT_KERNEL}', which needs "
            f"{estimate_device_bytes(n, m, DEFAULT_KERNEL) / 1e6:.2f} MB for the same score."
        )


def needleman_wunsch_gpu(seq1, seq2, kernel=None, use_shared=None, **kwargs):
    global _SHARED_KERNEL_ERROR

    if kernel is None:
        kernel = DEFAULT_KERNEL if use_shared is None else ("tiled" if use_shared else "diagonal")

    n, m = sequence_length(seq1), sequence_length(seq2)

    if kernel == "warp":
        if _SHARED_KERNEL_ERROR is None:
            try:
                return needleman_wunsch_cuda_warp(seq1, seq2, **kwargs)
            except DeviceMemoryTooSmall:
                raise
            except Exception as exc:
                if not SHARED_KERNEL_FALLBACK:
                    raise
                _SHARED_KERNEL_ERROR = f"{type(exc).__name__}: {exc}"
        kernel = "banded"

    if kernel == "banded":
        if _SHARED_KERNEL_ERROR is None:
            try:
                return needleman_wunsch_cuda_banded(seq1, seq2, **kwargs)
            except DeviceMemoryTooSmall:
                raise
            except Exception as exc:
                if not SHARED_KERNEL_FALLBACK:
                    raise
                _SHARED_KERNEL_ERROR = f"{type(exc).__name__}: {exc}"
        _guard_full_matrix(n, m, "diagonal")
        result = needleman_wunsch_cuda(seq1, seq2, **kwargs)
        result["kernel"] = "diagonal_global"
        result["shared_kernel_error"] = _SHARED_KERNEL_ERROR
        return result

    if kernel == "tiled":
        _guard_full_matrix(n, m, "tiled")
        if _SHARED_KERNEL_ERROR is None:
            try:
                return needleman_wunsch_cuda_shared(seq1, seq2, **kwargs)
            except Exception as exc:
                if not SHARED_KERNEL_FALLBACK:
                    raise
                _SHARED_KERNEL_ERROR = f"{type(exc).__name__}: {exc}"
        result = needleman_wunsch_cuda(seq1, seq2, **kwargs)
        result["kernel"] = "diagonal_global"
        result["shared_kernel_error"] = _SHARED_KERNEL_ERROR
        return result

    _guard_full_matrix(n, m, "diagonal")
    result = needleman_wunsch_cuda(seq1, seq2, **kwargs)
    result["kernel"] = "diagonal_global"
    return result


CHUNK_DIR = "chunks_1m"
CHUNK_FILENAMES = [f"genome_chunk_{i:02d}.npy" for i in range(10)]

SAMPLE_SIZE = None

INT_TO_BASE = {0: "A", 1: "C", 2: "G", 3: "T"}
UNKNOWN_BASE = "N"


def load_chunk_as_sequence(path, limit=None):
    arr = np.load(path, allow_pickle=True)
    if limit is not None:
        arr = arr[:limit]

    kind = arr.dtype.kind
    if kind in ("u", "i"):
        seq = "".join(INT_TO_BASE.get(int(x), UNKNOWN_BASE) for x in arr)
    elif kind == "S":
        seq = b"".join(arr.tolist()).decode("ascii")
    elif kind == "U":
        seq = "".join(arr.tolist())
    elif kind == "O":
        parts = [x.decode("ascii") if isinstance(x, bytes) else str(x) for x in arr.tolist()]
        seq = "".join(parts)
    else:
        raise ValueError(f"Unrecognized chunk dtype '{arr.dtype}' in {path}")
    return seq.upper()


@cuda.jit
def nw_tiled_banded_kernel(
    seq1_d, seq2_d, h_d, v_d, corner_in_d, corner_out_d,
    n, m, tile_diag, ti_lo, ti_hi, match, mismatch, gap,
):
    s_dp = cuda.shared.array(shape=(TILE_DIM_P1, TILE_DIM_P1), dtype=int32)
    s_s1 = cuda.shared.array(shape=TILE_DIM, dtype=int8)
    s_s2 = cuda.shared.array(shape=TILE_DIM, dtype=int8)

    ti = ti_lo + cuda.blockIdx.x
    if ti > ti_hi:
        return
    tj = tile_diag - ti

    i0 = ti * TILE_DIM + 1
    j0 = tj * TILE_DIM + 1
    rows = min(TILE_DIM, n - i0 + 1)
    cols = min(TILE_DIM, m - j0 + 1)
    if rows <= 0 or cols <= 0:
        return

    tx = cuda.threadIdx.x

    if tx == 0:
        if ti == 0:
            s_dp[0, 0] = (j0 - 1) * gap      # dp[0][j0-1] on the top boundary
        elif tj == 0:
            s_dp[0, 0] = (i0 - 1) * gap      # dp[i0-1][0] on the left boundary
        else:
            s_dp[0, 0] = corner_in_d[tj - 1]
    if tx < cols:
        s_dp[0, tx + 1] = h_d[j0 + tx]
        s_s2[tx] = seq2_d[j0 - 1 + tx]
    if tx < rows:
        s_dp[tx + 1, 0] = v_d[i0 + tx]
        s_s1[tx] = seq1_d[i0 - 1 + tx]

    cuda.syncthreads()

    for k in range(rows + cols - 1):
        c = k - tx
        if tx < rows and c >= 0 and c < cols:
            score = match if s_s1[tx] == s_s2[c] else mismatch
            diag_val = s_dp[tx, c] + score
            up_val = s_dp[tx, c + 1] + gap
            left_val = s_dp[tx + 1, c] + gap

            best = diag_val
            if up_val > best:
                best = up_val
            if left_val > best:
                best = left_val

            s_dp[tx + 1, c + 1] = best
        cuda.syncthreads()

    if tx < cols:
        h_d[j0 + tx] = s_dp[rows, tx + 1]
    if tx < rows:
        v_d[i0 + tx] = s_dp[tx + 1, cols]
    if tx == 0:
        corner_out_d[tj] = s_dp[rows, cols]


def needleman_wunsch_cuda_banded(
    seq1, seq2,
    match=1, mismatch=-1, gap=-2,
    progress_callback=None,
):
    n, m = sequence_length(seq1), sequence_length(seq2)

    seq1_codes = encode_sequence(seq1)
    seq2_codes = encode_sequence(seq2)

    n_ti = math.ceil(n / TILE_DIM)
    n_tj = math.ceil(m / TILE_DIM)

    h_init = (np.arange(m + 1, dtype=np.int64) * gap).astype(np.int32)  # dp[0][j]
    v_init = (np.arange(n + 1, dtype=np.int64) * gap).astype(np.int32)  # dp[i][0]
    corner_init = np.zeros(max(1, n_tj), dtype=np.int32)

    timings = {}

    t0 = time.perf_counter()
    seq1_d = cuda.to_device(seq1_codes)
    seq2_d = cuda.to_device(seq2_codes)
    h_d = cuda.to_device(h_init)
    v_d = cuda.to_device(v_init)
    corners = [cuda.to_device(corner_init.copy()) for _ in range(3)]
    cuda.synchronize()
    timings["host_to_device_s"] = time.perf_counter() - t0

    total_tile_diagonals = n_ti + n_tj - 1
    update_every = max(1, total_tile_diagonals // 100)

    t1 = time.perf_counter()
    for step, tile_diag in enumerate(range(total_tile_diagonals), start=1):
        ti_lo = max(0, tile_diag - (n_tj - 1))
        ti_hi = min(n_ti - 1, tile_diag)
        blocks = ti_hi - ti_lo + 1
        if blocks <= 0:
            continue

        corner_in = corners[(tile_diag + 1) % 3]
        corner_out = corners[tile_diag % 3]

        nw_tiled_banded_kernel[blocks, TILE_DIM](
            seq1_d, seq2_d, h_d, v_d, corner_in, corner_out,
            n, m, tile_diag, ti_lo, ti_hi, match, mismatch, gap,
        )

        if progress_callback is not None and (
            step % update_every == 0 or step == total_tile_diagonals
        ):
            cuda.synchronize()
            progress_callback(step, total_tile_diagonals)

    cuda.synchronize()
    timings["kernel_exec_s"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    try:
        score = int(h_d[m:m + 1].copy_to_host()[0])
    except Exception:
        score = int(h_d.copy_to_host()[m])
    timings["device_to_host_s"] = time.perf_counter() - t2

    timings["total_s"] = sum(timings.values())

    return {"score": score, "n": n, "m": m, "timings": timings, "kernel": "tiled_banded"}


TILE_ROWS = 32
TILE_ROWS_P1 = 33
TILE_COLS = 128
WARPS_PER_BLOCK = 4
BLOCK_THREADS = 128
assert TILE_ROWS_P1 == TILE_ROWS + 1
assert BLOCK_THREADS == TILE_ROWS * WARPS_PER_BLOCK


@cuda.jit
def nw_warp_tile_kernel(
    seq1_d, seq2_d, h_d, v_d, corner_in_d, corner_out_d,
    n, m, tile_diag, ti_lo, n_tiles_on_diag, match, mismatch, gap,
):
    s_buf = cuda.shared.array(shape=(WARPS_PER_BLOCK, 3, TILE_ROWS_P1), dtype=int32)
    s_top = cuda.shared.array(shape=(WARPS_PER_BLOCK, TILE_COLS), dtype=int32)
    s_out = cuda.shared.array(shape=(WARPS_PER_BLOCK, TILE_COLS), dtype=int32)
    s_left = cuda.shared.array(shape=(WARPS_PER_BLOCK, TILE_ROWS), dtype=int32)
    s_outc = cuda.shared.array(shape=(WARPS_PER_BLOCK, TILE_ROWS), dtype=int32)
    s_s1 = cuda.shared.array(shape=(WARPS_PER_BLOCK, TILE_ROWS), dtype=int8)
    s_s2 = cuda.shared.array(shape=(WARPS_PER_BLOCK, TILE_COLS), dtype=int8)

    tid = cuda.threadIdx.x
    wid = tid // TILE_ROWS      # which tile this warp owns
    lane = tid % TILE_ROWS      # which tile row this lane owns

    slot = cuda.blockIdx.x * WARPS_PER_BLOCK + wid
    if slot >= n_tiles_on_diag:
        return                  # whole warp exits together

    ti = ti_lo + slot
    tj = tile_diag - ti

    i0 = ti * TILE_ROWS + 1
    j0 = tj * TILE_COLS + 1
    rows = min(TILE_ROWS, n - i0 + 1)
    cols = min(TILE_COLS, m - j0 + 1)
    if rows <= 0 or cols <= 0:
        return

    if lane < rows:
        s_s1[wid, lane] = seq1_d[i0 - 1 + lane]
        s_left[wid, lane] = v_d[i0 + lane]
    c = lane
    while c < cols:
        s_top[wid, c] = h_d[j0 + c]
        s_s2[wid, c] = seq2_d[j0 - 1 + c]
        c += TILE_ROWS

    if lane == 0:
        if ti == 0:
            s_buf[wid, 1, 0] = (j0 - 1) * gap
        elif tj == 0:
            s_buf[wid, 1, 0] = (i0 - 1) * gap
        else:
            s_buf[wid, 1, 0] = corner_in_d[tj - 1]

    cuda.syncwarp()

    for w in range(-1, rows + cols - 1):
        ww = (w + 3) % 3
        w1 = (w + 2) % 3
        w2 = (w + 1) % 3

        if lane == 0 and w + 1 < cols:
            s_buf[wid, ww, 0] = s_top[wid, w + 1]
        if lane == w + 1 and lane < rows:
            s_buf[wid, ww, lane + 1] = s_left[wid, lane]
        cuda.syncwarp()

        cc = w - lane
        if lane < rows and cc >= 0 and cc < cols:
            score = match if s_s1[wid, lane] == s_s2[wid, cc] else mismatch
            best = s_buf[wid, w2, lane] + score
            up_val = s_buf[wid, w1, lane] + gap
            left_val = s_buf[wid, w1, lane + 1] + gap
            if up_val > best:
                best = up_val
            if left_val > best:
                best = left_val
            s_buf[wid, ww, lane + 1] = best
            if lane == rows - 1:
                s_out[wid, cc] = best
            if cc == cols - 1:
                s_outc[wid, lane] = best
        cuda.syncwarp()

    c = lane
    while c < cols:
        h_d[j0 + c] = s_out[wid, c]
        c += TILE_ROWS
    if lane < rows:
        v_d[i0 + lane] = s_outc[wid, lane]
    if lane == 0:
        corner_out_d[tj] = s_out[wid, cols - 1]


def needleman_wunsch_cuda_warp(
    seq1, seq2,
    match=1, mismatch=-1, gap=-2,
    progress_callback=None,
):
    n, m = sequence_length(seq1), sequence_length(seq2)

    seq1_codes = encode_sequence(seq1)
    seq2_codes = encode_sequence(seq2)

    n_ti = math.ceil(n / TILE_ROWS)
    n_tj = math.ceil(m / TILE_COLS)

    h_init = (np.arange(m + 1, dtype=np.int64) * gap).astype(np.int32)
    v_init = (np.arange(n + 1, dtype=np.int64) * gap).astype(np.int32)
    corner_init = np.zeros(max(1, n_tj), dtype=np.int32)

    timings = {}

    t0 = time.perf_counter()
    seq1_d = cuda.to_device(seq1_codes)
    seq2_d = cuda.to_device(seq2_codes)
    h_d = cuda.to_device(h_init)
    v_d = cuda.to_device(v_init)
    corners = [cuda.to_device(corner_init.copy()) for _ in range(3)]
    cuda.synchronize()
    timings["host_to_device_s"] = time.perf_counter() - t0

    total_tile_diagonals = n_ti + n_tj - 1
    update_every = max(1, total_tile_diagonals // 100)

    t1 = time.perf_counter()
    for step, tile_diag in enumerate(range(total_tile_diagonals), start=1):
        ti_lo = max(0, tile_diag - (n_tj - 1))
        ti_hi = min(n_ti - 1, tile_diag)
        n_tiles = ti_hi - ti_lo + 1
        if n_tiles <= 0:
            continue

        blocks = math.ceil(n_tiles / WARPS_PER_BLOCK)
        nw_warp_tile_kernel[blocks, BLOCK_THREADS](
            seq1_d, seq2_d, h_d, v_d,
            corners[(tile_diag + 1) % 3], corners[tile_diag % 3],
            n, m, tile_diag, ti_lo, n_tiles, match, mismatch, gap,
        )

        if progress_callback is not None and (
            step % update_every == 0 or step == total_tile_diagonals
        ):
            cuda.synchronize()
            progress_callback(step, total_tile_diagonals)

    cuda.synchronize()
    timings["kernel_exec_s"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    try:
        score = int(h_d[m:m + 1].copy_to_host()[0])
    except Exception:
        score = int(h_d.copy_to_host()[m])
    timings["device_to_host_s"] = time.perf_counter() - t2

    timings["total_s"] = sum(timings.values())

    return {"score": score, "n": n, "m": m, "timings": timings, "kernel": "warp_tile"}


def occupancy_report(n, m, kernel="warp"):
    if kernel == "warp":
        tr, tc, warps = TILE_ROWS, TILE_COLS, WARPS_PER_BLOCK
        smem = warps * (3 * TILE_ROWS_P1 * 4 + TILE_COLS * 4 * 2 + TILE_ROWS * 4 * 2
                        + TILE_ROWS + TILE_COLS)
    else:
        tr, tc, warps = TILE_DIM, TILE_DIM, 1
        smem = TILE_DIM_P1 * TILE_DIM_P1 * 4 + 2 * TILE_DIM
    threads = tr * warps
    steps = tr + tc - 1
    busy = sum(sum(1 for t in range(tr) if 0 <= k - t < tc) for k in range(steps))
    n_ti = math.ceil(n / tr) if n else 0
    n_tj = math.ceil(m / tc) if m else 0
    return {
        "tile": f"{tr}x{tc}",
        "warps_per_block": warps,
        "threads_per_block": threads,
        "shared_per_block_b": smem,
        "lane_efficiency": busy / (steps * tr) if steps else 0.0,
        "launches_per_pair": max(0, n_ti + n_tj - 1),
    }


def estimate_device_bytes(n, m, kernel="warp"):
    seqs = n + m
    if kernel == "warp":
        return seqs + 4 * (n + 1) + 4 * (m + 1) + 3 * 4 * max(1, math.ceil(m / TILE_COLS))
    if kernel == "banded":
        return seqs + 4 * (n + 1) + 4 * (m + 1) + 3 * 4 * max(1, math.ceil(m / TILE_DIM))
    return seqs + 4 * (n + 1) * (m + 1)


def check_device_capacity(n, m, kernel="warp", headroom=0.90):
    needed = estimate_device_bytes(n, m, kernel)
    free = None
    try:
        free, _total = cuda.current_context().get_memory_info()
    except Exception:
        try:
            rows = _nvidia_smi_rows(["memory.free"])
            if rows:
                free = float(rows[0]["memory.free"]) * 1024 * 1024
        except Exception:
            free = None
    if free is None:
        return True, needed, None
    return needed <= free * headroom, needed, int(free)


def load_chunk_as_codes(path, limit=None):
    arr = np.load(path, allow_pickle=True, mmap_mode="r")
    if limit is not None:
        arr = arr[:limit]
    arr = np.asarray(arr)

    kind = arr.dtype.kind
    if kind in ("u", "i"):
        codes = arr.astype(np.int8, copy=True)
        np.clip(codes, 0, UNKNOWN_CODE, out=codes)
        return codes
    if kind == "S":
        raw = np.frombuffer(arr.tobytes(), dtype=np.uint8)
        return _CODE_LUT[raw]
    return encode_sequence(load_chunk_as_sequence(path, limit))


def sequence_length(seq):
    return int(seq.shape[0]) if isinstance(seq, np.ndarray) else len(seq)


def align_sequence_pairs_gpu(sequences, progress_callback=None, kernel=None, use_shared=None):
    pair_results = []
    n_pairs = len(sequences) - 1
    for i in range(n_pairs):
        def cb(done_diag, total_diag, _i=i):
            if progress_callback is not None:
                progress_callback(_i, n_pairs, done_diag, total_diag)

        result = needleman_wunsch_gpu(
            sequences[i], sequences[i + 1],
            kernel=kernel, use_shared=use_shared,
            progress_callback=cb if progress_callback is not None else None,
        )
        result["pair"] = f"{i}-{i+1}"
        pair_results.append(result)

    return pair_results


def align_all_chunks_gpu(chunk_dir, chunk_filenames, sample_size=None,
                         progress_callback=None, kernel=None, use_shared=None):
    import os as _os
    paths = [_os.path.join(chunk_dir, f) for f in chunk_filenames]
    sequences = [load_chunk_as_codes(p, sample_size) for p in paths]
    return align_sequence_pairs_gpu(
        sequences, progress_callback, kernel=kernel, use_shared=use_shared
    )


FORCE_GPU_COUNT = None


def _nvidia_smi_rows(fields, timeout=3):
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + ",".join(fields),
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout,
        )
    except Exception:
        return []

    if out.returncode != 0 or not out.stdout.strip():
        return []

    rows = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == len(fields):
            rows.append(dict(zip(fields, parts)))
    return rows


def _cudasim_active():
    return os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"


def detect_gpu_count():
    if FORCE_GPU_COUNT is not None:
        return int(FORCE_GPU_COUNT)

    if _cudasim_active():
        return 1

    try:
        if cuda.is_available():
            n = len(list(cuda.list_devices()))
            if n > 0:
                return n
    except Exception:
        pass

    return len(_nvidia_smi_rows(["index"]))


def list_gpu_devices():
    if _cudasim_active():
        return [{
            "index": 0, "name": "CUDA Simulator",
            "memory_total_gb": None, "memory_free_gb": None, "utilization_pct": None,
        }]

    smi = _nvidia_smi_rows(
        ["index", "name", "memory.total", "memory.free", "utilization.gpu"]
    )

    devices = []
    for row in smi:
        def _num(key):
            try:
                return float(row[key])
            except (KeyError, ValueError):
                return None

        total_mb, free_mb = _num("memory.total"), _num("memory.free")
        devices.append({
            "index": int(row["index"]) if row["index"].isdigit() else len(devices),
            "name": row.get("name") or "GPU",
            "memory_total_gb": total_mb / 1024 if total_mb is not None else None,
            "memory_free_gb": free_mb / 1024 if free_mb is not None else None,
            "utilization_pct": _num("utilization.gpu"),
        })

    if devices:
        return devices

    try:
        for dev in cuda.list_devices():
            name = getattr(dev, "name", "GPU")
            devices.append({
                "index": getattr(dev, "id", len(devices)),
                "name": name.decode() if isinstance(name, bytes) else str(name),
                "memory_total_gb": None, "memory_free_gb": None, "utilization_pct": None,
            })
    except Exception:
        pass

    return devices


def gpu_status_info():
    devices = list_gpu_devices()
    count = detect_gpu_count()

    info = {
        "available": False,
        "simulator": _cudasim_active(),
        "count": count,
        "devices": devices,
        "forced_count": FORCE_GPU_COUNT is not None,
        "device_name": None,
        "memory_free_gb": None,
        "memory_total_gb": None,
        "utilization_pct": None,
    }

    try:
        info["available"] = bool(cuda.is_available())
    except Exception:
        info["available"] = False

    try:
        if info["available"]:
            dev = cuda.get_current_device()
            name = dev.name
            info["device_name"] = name.decode() if isinstance(name, bytes) else str(name)
    except Exception:
        pass

    if info["device_name"] is None and devices:
        info["device_name"] = devices[0]["name"]

    totals = [d["memory_total_gb"] for d in devices if d["memory_total_gb"] is not None]
    frees = [d["memory_free_gb"] for d in devices if d["memory_free_gb"] is not None]
    if totals:
        info["memory_total_gb"] = sum(totals)
    if frees:
        info["memory_free_gb"] = sum(frees)

    if info["memory_total_gb"] is None:
        try:
            free_b, total_b = cuda.current_context().get_memory_info()
            if math.isfinite(free_b) and math.isfinite(total_b):
                info["memory_free_gb"] = free_b / 1e9
                info["memory_total_gb"] = total_b / 1e9
        except Exception:
            pass

    utils = [d["utilization_pct"] for d in devices if d["utilization_pct"] is not None]
    if utils:
        info["utilization_pct"] = sum(utils) / len(utils)

    return info


def create_dask_cluster(n_workers=None, threads_per_worker=1):
    if n_workers is None:
        n_workers = detect_gpu_count()
    n_workers = max(1, int(n_workers))

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        processes=True,
        dashboard_address=None,
    )
    return Client(cluster), cluster


def parse_chunks_dask(
    chunk_dir, chunk_filenames, sample_size=None,
    client=None, progress_callback=None, as_codes=True,
):
    paths = [os.path.join(chunk_dir, f) for f in chunk_filenames]
    return parse_paths_dask(paths, sample_size, client, progress_callback, as_codes)


def parse_paths_dask(paths, sample_size=None, client=None, progress_callback=None,
                     as_codes=True):
    paths = [str(p) for p in paths]
    owns_client = client is None
    cluster = None
    if owns_client:
        client, cluster = create_dask_cluster()

    try:
        loader = load_chunk_as_codes if as_codes else load_chunk_as_sequence
        futures = [client.submit(loader, p, sample_size, pure=False) for p in paths]
        index_of = {f.key: i for i, f in enumerate(futures)}

        sequences = [None] * len(futures)
        done = 0
        for fut in dask_as_completed(futures):
            i = index_of[fut.key]
            sequences[i] = fut.result()
            done += 1
            if progress_callback is not None:
                progress_callback(done, len(futures), os.path.basename(paths[i]))

        return sequences
    finally:
        if owns_client:
            client.close()
            if cluster is not None:
                cluster.close()


_PINNED_GPU_ID = None  # per worker process: set once, never thrashed


def _pin_process_to_gpu(gpu_id):
    global _PINNED_GPU_ID
    if _PINNED_GPU_ID == gpu_id:
        return _PINNED_GPU_ID
    try:
        cuda.select_device(gpu_id)
        _PINNED_GPU_ID = gpu_id
    except Exception:
        _PINNED_GPU_ID = None  # single-GPU / simulator: nothing to select
    return _PINNED_GPU_ID


def _align_pair_on_gpu(seq1, seq2, gpu_id, pair_label, kernel=None):
    _pin_process_to_gpu(gpu_id)

    t0 = time.perf_counter()
    result = needleman_wunsch_gpu(seq1, seq2, kernel=kernel)
    result["pair"] = pair_label
    result["gpu_id"] = gpu_id
    result["worker"] = _current_worker_address()
    if shared_kernel_error() is not None:
        result["shared_kernel_error"] = shared_kernel_error()
    result["wall_s"] = time.perf_counter() - t0
    return result


def _current_worker_address():
    try:
        from dask.distributed import get_worker
        return get_worker().address
    except Exception:
        return None


def worker_gpu_map(client, n_gpus):
    try:
        addrs = sorted(client.scheduler_info()["workers"].keys())
    except Exception:
        return []
    n_gpus = max(1, int(n_gpus))
    return [(addr, idx % n_gpus) for idx, addr in enumerate(addrs)]


def align_sequence_pairs_gpu_dask(
    sequences, client=None, n_gpus=None, progress_callback=None, kernel=None,
):
    if n_gpus is None:
        n_gpus = detect_gpu_count()
    n_gpus = max(1, int(n_gpus))

    owns_client = client is None
    cluster = None
    if owns_client:
        client, cluster = create_dask_cluster(n_gpus)

    try:
        n_pairs = len(sequences) - 1
        mapping = worker_gpu_map(client, n_gpus)

        futures = []
        for i in range(n_pairs):
            if mapping:
                addr, gpu_id = mapping[i % len(mapping)]
                submit_kwargs = {"workers": [addr], "allow_other_workers": False}
            else:  # cannot see the worker list: tag the device and let Dask place it
                gpu_id = i % n_gpus
                submit_kwargs = {}

            futures.append(client.submit(
                _align_pair_on_gpu,
                sequences[i], sequences[i + 1], gpu_id, f"{i}-{i+1}", kernel,
                pure=False,
                **submit_kwargs,
            ))

        index_of = {f.key: i for i, f in enumerate(futures)}
        results = [None] * n_pairs
        done = 0
        for fut in dask_as_completed(futures):
            i = index_of[fut.key]
            results[i] = fut.result()
            done += 1
            if progress_callback is not None:
                progress_callback(done, n_pairs, results[i]["pair"])

        return results, n_gpus
    finally:
        if owns_client:
            client.close()
            if cluster is not None:
                cluster.close()


def align_all_chunks_gpu_dask(chunk_dir, chunk_filenames, sample_size=None,
                              n_gpus=None, kernel=None):
    if n_gpus is None:
        n_gpus = detect_gpu_count()
    n_gpus = max(1, int(n_gpus))

    client, cluster = create_dask_cluster(n_gpus)
    try:
        sequences = parse_chunks_dask(chunk_dir, chunk_filenames, sample_size, client=client)
        return align_sequence_pairs_gpu_dask(
            sequences, client=client, n_gpus=n_gpus, kernel=kernel
        )
    finally:
        client.close()
        cluster.close()


if __name__ == "__main__":
    print(f"CUDA available (real GPU): {cuda.is_available()}")
    print(f"CUDA simulator active    : {_cudasim_active()}")

    n_gpus = detect_gpu_count()
    print(f"GPUs detected            : {n_gpus}"
          + (" (forced via FORCE_GPU_COUNT)" if FORCE_GPU_COUNT is not None else ""))
    for d in list_gpu_devices():
        mem = (f"{d['memory_free_gb']:.2f}/{d['memory_total_gb']:.2f} GB free"
               if d["memory_total_gb"] is not None else "memory N/A")
        util = f"{d['utilization_pct']:.0f}% util" if d["utilization_pct"] is not None else "util N/A"
        print(f"  [{d['index']}] {d['name']} - {mem}, {util}")
    print()

    print(f"Loading chunks from '{CHUNK_DIR}' "
          + (f"(sampling first {SAMPLE_SIZE:,} bp of each)" if SAMPLE_SIZE else "(full length)"))

    def _report(label, rows):
        print(f"\n[{label}]")
        print(f"{'Pair':>8} | {'Length':>8} | {'H2D (ms)':>9} | {'Kernel (ms)':>11} | "
              f"{'D2H (ms)':>9} | {'Total (ms)':>10} | {'Score':>8}")
        print("-" * 78)
        for r in rows:
            t = r["timings"]
            print(f"{r['pair']:>8} | {r['n']:>8,} | {t['host_to_device_s']*1000:>9.3f} | "
                  f"{t['kernel_exec_s']*1000:>11.3f} | {t['device_to_host_s']*1000:>9.3f} | "
                  f"{t['total_s']*1000:>10.3f} | {r['score']:>8}")
        total = sum(r["timings"]["total_s"] for r in rows) * 1000
        print("-" * 78)
        print(f"Pairs aligned : {len(rows)}")
        print(f"Total time    : {total:.3f} ms")
        print(f"Avg time/pair : {total/len(rows):.3f} ms")
        return total

    probe = load_chunk_as_codes(os.path.join(CHUNK_DIR, CHUNK_FILENAMES[0]), SAMPLE_SIZE)
    bp = sequence_length(probe)
    full_gb = estimate_device_bytes(bp, bp, "tiled") / 1e9
    band_mb = estimate_device_bytes(bp, bp, "banded") / 1e6
    fits_full, _need, free = check_device_capacity(bp, bp, "tiled")
    print(f"\nChunk length  : {bp:,} bp")
    print(f"Full DP matrix: {full_gb:,.2f} GB VRAM per pair "
          + ("(fits)" if fits_full else "(DOES NOT FIT - full-matrix kernels skipped)"))
    print(f"Banded        : {band_mb:,.2f} MB VRAM per pair")
    if free is not None:
        print(f"Free VRAM     : {free / 1e9:.2f} GB")
    print(f"DP cells/pair : {bp * bp:,}")

    results = align_all_chunks_gpu(CHUNK_DIR, CHUNK_FILENAMES, SAMPLE_SIZE, kernel="warp")
    occ = occupancy_report(bp, bp, "warp")
    print(f"Tile          : {occ['tile']}, {occ['warps_per_block']} warps/block "
          f"({occ['threads_per_block']} threads), {occ['shared_per_block_b']/1024:.1f} KB shared/block")
    print(f"Lane use      : {occ['lane_efficiency']*100:.1f}%   launches/pair: {occ['launches_per_pair']:,}")
    banded_ms = _report(f"warp-tile kernel ({occ['tile']}), O(n+m) VRAM", results)

    if shared_kernel_error() is not None:
        print(f"\n!! Shared-memory kernel unavailable, fell back: {shared_kernel_error()}")

    if fits_full:
        naive = align_all_chunks_gpu(CHUNK_DIR, CHUNK_FILENAMES, SAMPLE_SIZE, kernel="diagonal")
        naive_ms = _report("diagonal kernel - global memory, O(n*m) VRAM", naive)
        mismatched = [(a["pair"], a["score"], b["score"])
                      for a, b in zip(naive, results) if a["score"] != b["score"]]
        print(f"\nScores identical across kernels: {not mismatched}"
              + (f" -- MISMATCHES: {mismatched}" if mismatched else ""))
        if banded_ms > 0:
            print(f"Banded vs diagonal speedup     : {naive_ms/banded_ms:.2f}x")
    else:
        print("\nSkipped the diagonal-kernel comparison: it would need "
              f"{full_gb:,.2f} GB of VRAM at this chunk size.")

    print(f"\n\nDask pipeline (parse + align) - using {max(1, n_gpus)} worker(s), "
          f"one GPU pinned per worker")

    dask_results, n_gpus_used = align_all_chunks_gpu_dask(
        CHUNK_DIR, CHUNK_FILENAMES, SAMPLE_SIZE, kernel="warp"
    )

    print(f"\n{'Pair':>8} | {'GPU':>4} | {'Wall (ms)':>10} | {'Kernel (ms)':>11} | "
          f"{'Score':>8} | {'Worker':<24}")
    print("-" * 82)
    for r in dask_results:
        print(f"{r['pair']:>8} | {r['gpu_id']:>4} | {r['wall_s']*1000:>10.2f} | "
              f"{r['timings']['kernel_exec_s']*1000:>11.2f} | {r['score']:>8} | "
              f"{str(r.get('worker') or '-'):<24}")

    per_gpu_counts, per_gpu_time = {}, {}
    for r in dask_results:
        per_gpu_counts[r["gpu_id"]] = per_gpu_counts.get(r["gpu_id"], 0) + 1
        per_gpu_time[r["gpu_id"]] = per_gpu_time.get(r["gpu_id"], 0.0) + r["wall_s"]

    print("-" * 82)
    print(f"Pairs aligned : {len(dask_results)} across {n_gpus_used} GPU(s)")
    for gpu_id in sorted(per_gpu_counts):
        print(f"  GPU {gpu_id}: {per_gpu_counts[gpu_id]} pairs, "
              f"{per_gpu_time[gpu_id]*1000:.2f} ms total")
    spread = max(per_gpu_counts.values()) - min(per_gpu_counts.values()) if per_gpu_counts else 0
    print(f"Balance spread: {spread} pair(s) "
          + ("(even)" if spread <= 1 else "(UNEVEN - check worker pinning)"))