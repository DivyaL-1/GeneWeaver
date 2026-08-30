import os
import math
import time
import numpy as np
from numba import cuda, int8, int32
from dask.distributed import Client, LocalCluster


BASE_TO_CODE = {"A": 0, "C": 1, "G": 2, "T": 3}
UNKNOWN_CODE = 4


def encode_sequence(seq_str):
    """ACGTN string -> int8 numpy array of codes (A=0,C=1,G=2,T=3,N=4)."""
    return np.array(
        [BASE_TO_CODE.get(ch, UNKNOWN_CODE) for ch in seq_str], dtype=np.int8
    )


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
    n, m = len(seq1_str), len(seq2_str)
    row_stride = m + 1

    seq1_codes = encode_sequence(seq1_str)
    seq2_codes = encode_sequence(seq2_str)

    dp_init = np.zeros((n + 1) * (m + 1), dtype=np.int32)
    for i in range(n + 1):
        dp_init[i * row_stride] = i * gap
    for j in range(m + 1):
        dp_init[j] = j * gap

    timings = {}

    # ---- Host -> Device transfer ----
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

    # ---- Device -> Host transfer ----
    t2 = time.perf_counter()
    dp_result = dp_d.copy_to_host()
    timings["device_to_host_s"] = time.perf_counter() - t2

    score = int(dp_result[n * row_stride + m])
    timings["total_s"] = sum(timings.values())

    return {"score": score, "n": n, "m": m, "timings": timings}


# >>> SET THESE TWO VARIABLES TO YOUR CHUNK FILE LOCATION <<<
CHUNK_DIR = "chunks_1m"
CHUNK_FILENAMES = [f"genome_chunk_{i:02d}.npy" for i in range(10)]

SAMPLE_SIZE = 5000

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


def align_sequence_pairs_gpu(sequences, progress_callback=None):
    pair_results = []
    n_pairs = len(sequences) - 1
    for i in range(n_pairs):
        def cb(done_diag, total_diag, _i=i):
            if progress_callback is not None:
                progress_callback(_i, n_pairs, done_diag, total_diag)

        result = needleman_wunsch_cuda(
            sequences[i], sequences[i + 1],
            progress_callback=cb if progress_callback is not None else None,
        )
        result["pair"] = f"{i}-{i+1}"
        pair_results.append(result)

    return pair_results


def align_all_chunks_gpu(chunk_dir, chunk_filenames, sample_size=None, progress_callback=None):
    import os as _os
    paths = [_os.path.join(chunk_dir, f) for f in chunk_filenames]
    sequences = [load_chunk_as_sequence(p, sample_size) for p in paths]
    return align_sequence_pairs_gpu(sequences, progress_callback)


def gpu_status_info():
    info = {
        "available": cuda.is_available(),
        "simulator": os.environ.get("NUMBA_ENABLE_CUDASIM") == "1",
        "device_name": None,
        "memory_free_gb": None,
        "memory_total_gb": None,
        "utilization_pct": None,
    }

    try:
        if info["available"]:
            dev = cuda.get_current_device()
            name = dev.name
            info["device_name"] = name.decode() if isinstance(name, bytes) else str(name)
    except Exception:
        pass

    try:
        free_b, total_b = cuda.current_context().get_memory_info()
        if math.isfinite(free_b) and math.isfinite(total_b):
            info["memory_free_gb"] = free_b / 1e9
            info["memory_total_gb"] = total_b / 1e9
    except Exception:
        pass

    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode == 0 and out.stdout.strip():
            info["utilization_pct"] = float(out.stdout.strip().splitlines()[0])
    except Exception:
        pass

    return info


FORCE_GPU_COUNT = None


def detect_gpu_count():
    """Real GPU count via Numba, unless overridden above for testing."""
    if FORCE_GPU_COUNT is not None:
        return FORCE_GPU_COUNT
    try:
        return max(1, len(cuda.list_devices()))
    except Exception:
        return 1


def _align_pair_on_gpu(seq1, seq2, gpu_id, pair_label):
    try:
        cuda.select_device(gpu_id)
    except Exception:
        pass  # single-GPU / simulator: nothing meaningful to select

    t0 = time.perf_counter()
    result = needleman_wunsch_cuda(seq1, seq2)
    result["pair"] = pair_label
    result["gpu_id"] = gpu_id
    result["wall_s"] = time.perf_counter() - t0
    return result


def align_all_chunks_gpu_dask(chunk_dir, chunk_filenames, sample_size=None, n_gpus=None):
    n_gpus = max(1, n_gpus if n_gpus is not None else detect_gpu_count())

    paths = [os.path.join(chunk_dir, f) for f in chunk_filenames]
    sequences = [load_chunk_as_sequence(p, sample_size) for p in paths]
    n_pairs = len(sequences) - 1

    cluster = LocalCluster(n_workers=n_gpus, threads_per_worker=1, processes=True)
    client = Client(cluster)
    try:
        futures = []
        for i in range(n_pairs):
            gpu_id = i % n_gpus  # round robin -> most even possible split
            fut = client.submit(
                _align_pair_on_gpu,
                sequences[i], sequences[i + 1], gpu_id, f"{i}-{i+1}",
                pure=False,
            )
            futures.append(fut)
        results = client.gather(futures)
    finally:
        client.close()
        cluster.close()

    return results, n_gpus


if __name__ == "__main__":
    print(f"CUDA available (real GPU): {cuda.is_available()}")
    print(f"CUDA simulator active    : {os.environ.get('NUMBA_ENABLE_CUDASIM') == '1'}\n")

    print(f"Loading chunks from '{CHUNK_DIR}' "
          + (f"(sampling first {SAMPLE_SIZE:,} bp of each)" if SAMPLE_SIZE else "(full length)"))

    results = align_all_chunks_gpu(CHUNK_DIR, CHUNK_FILENAMES, SAMPLE_SIZE)

    print(f"\n{'Pair':>8} | {'Length':>8} | {'H2D (ms)':>9} | {'Kernel (ms)':>11} | "
          f"{'D2H (ms)':>9} | {'Total (ms)':>10} | {'Score':>8}")
    print("-" * 78)
    for r in results:
        t = r["timings"]
        print(f"{r['pair']:>8} | {r['n']:>8,} | {t['host_to_device_s']*1000:>9.3f} | "
              f"{t['kernel_exec_s']*1000:>11.3f} | {t['device_to_host_s']*1000:>9.3f} | "
              f"{t['total_s']*1000:>10.3f} | {r['score']:>8}")

    total_ms = sum(r["timings"]["total_s"] for r in results) * 1000
    print("-" * 78)
    print(f"Pairs aligned : {len(results)}")
    print(f"Total time    : {total_ms:.3f} ms")
    print(f"Avg time/pair : {total_ms/len(results):.3f} ms")

    n_gpus = detect_gpu_count()
    print(f"\n\nDask multi-GPU pipeline — detected {n_gpus} GPU(s)"
          + (" (forced via FORCE_GPU_COUNT)" if FORCE_GPU_COUNT is not None else ""))

    dask_results, n_gpus = align_all_chunks_gpu_dask(CHUNK_DIR, CHUNK_FILENAMES, SAMPLE_SIZE)

    print(f"\n{'Pair':>8} | {'GPU':>4} | {'Wall (ms)':>10} | {'Kernel (ms)':>11} | {'Score':>8}")
    print("-" * 55)
    for r in dask_results:
        print(f"{r['pair']:>8} | {r['gpu_id']:>4} | {r['wall_s']*1000:>10.2f} | "
              f"{r['timings']['kernel_exec_s']*1000:>11.2f} | {r['score']:>8}")

    per_gpu_counts, per_gpu_time = {}, {}
    for r in dask_results:
        per_gpu_counts[r["gpu_id"]] = per_gpu_counts.get(r["gpu_id"], 0) + 1
        per_gpu_time[r["gpu_id"]] = per_gpu_time.get(r["gpu_id"], 0.0) + r["wall_s"]

    print("-" * 55)
    print(f"Pairs aligned : {len(dask_results)} across {n_gpus} GPU(s)")
    for gpu_id in sorted(per_gpu_counts):
        print(f"  GPU {gpu_id}: {per_gpu_counts[gpu_id]} pairs, "
              f"{per_gpu_time[gpu_id]*1000:.2f} ms total")