
import numpy as np

if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack

import os
import time
import numpy as np


# SET THESE TWO VARIABLES TO YOUR CHUNK FILE LOCATION
CHUNK_DIR = "data//chunks"                                          # directory holding the chunk files
CHUNK_FILENAMES = [f"chunk_{i:06d}.npy" for i in range(1,11)]  # the 10 chunk filenames

SAMPLE_SIZE = 5000  #(None = full length) Current Max Threshold 37000 chunk length (tested on 16GB memory)

INT_TO_BASE = {0: "A", 1: "C", 2: "G", 3: "T"}
UNKNOWN_BASE = "N"

def load_chunk_as_sequence(path, limit=None):
    """
    Load a .npy genome chunk and return it as an ACGTN string.
    """
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
        parts = []
        for x in arr.tolist() if arr.ndim else [arr.item()]:
            parts.append(x.decode("ascii") if isinstance(x, bytes) else str(x))
        seq = "".join(parts)
    else:
        raise ValueError(f"Unrecognized chunk dtype '{arr.dtype}' in {path}")

    return seq.upper()


def needleman_wunsch(seq1, seq2, match=1, mismatch=-1, gap=-2):
    n, m = len(seq1), len(seq2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i * gap
    for j in range(1, m + 1):
        dp[0][j] = j * gap

    for i in range(1, n + 1):
        row = dp[i]
        prev_row = dp[i - 1]
        s1_char = seq1[i - 1]
        for j in range(1, m + 1):
            score = match if s1_char == seq2[j - 1] else mismatch
            diag = prev_row[j - 1] + score
            up = prev_row[j] + gap
            left = row[j - 1] + gap
            row[j] = diag if diag >= up and diag >= left else (up if up >= left else left)

    return dp[n][m]


def align_sequence_pairs_cpu(sequences, progress_callback=None):
    pair_results = []
    n_pairs = len(sequences) - 1
    for i in range(n_pairs):
        seq1, seq2 = sequences[i], sequences[i + 1]
        start = time.perf_counter()
        score = needleman_wunsch(seq1, seq2)
        elapsed = time.perf_counter() - start

        cells = len(seq1) * len(seq2)
        pair_results.append({
            "pair": f"{i}-{i+1}",
            "score": score,
            "n": len(seq1),
            "m": len(seq2),
            "elapsed_s": elapsed,
            "cells": cells,
            "cells_per_sec": cells / elapsed if elapsed > 0 else float("inf"),
        })

        if progress_callback is not None:
            progress_callback(i, n_pairs)

    return pair_results


def align_all_chunks_cpu(chunk_dir, chunk_filenames, sample_size=None, progress_callback=None):
    """Load every chunk file and run the CPU baseline on each consecutive pair."""
    paths = [os.path.join(chunk_dir, f) for f in chunk_filenames]
    sequences = [load_chunk_as_sequence(p, sample_size) for p in paths]
    return align_sequence_pairs_cpu(sequences, progress_callback)


if __name__ == "__main__":
    print(f"Loading {len(CHUNK_FILENAMES)} chunks from '{CHUNK_DIR}'"
          + (f", sampling first {SAMPLE_SIZE:,} bp of each" if SAMPLE_SIZE else " (full length)"))

    pair_results = align_all_chunks_cpu(CHUNK_DIR, CHUNK_FILENAMES, SAMPLE_SIZE)

    full_len = 1_000_000  # bp per chunk, per the stated chunking scheme
    print(f"\n{'Pair':>10} | {'Length':>8} | {'Time (s)':>10} | {'Cells/sec':>13} | {'Score':>8}")
    print("-" * 62)
    for r in pair_results:
        print(f"{r['pair']:>10} | {r['n']:>8,} | {r['elapsed_s']:>10.4f} | "
              f"{r['cells_per_sec']:>13,.0f} | {r['score']:>8}")

    total_time = sum(r["elapsed_s"] for r in pair_results)
    avg_time = total_time / len(pair_results)
    avg_throughput = sum(r["cells_per_sec"] for r in pair_results) / len(pair_results)

    print("-" * 62)
    print(f"Pairs aligned       : {len(pair_results)}")
    print(f"Total time          : {total_time:.4f} s")
    print(f"Average time/pair   : {avg_time:.4f} s")
    print(f"Average throughput  : {avg_throughput:,.0f} cells/sec")

    if SAMPLE_SIZE is not None and SAMPLE_SIZE < full_len:
        full_cells = full_len * full_len
        est_per_pair = full_cells / avg_throughput
        est_total = est_per_pair * len(pair_results)
        print(f"\nExtrapolated to full {full_len:,} bp chunks:")
        print(f"  ~{est_per_pair:,.0f} s (~{est_per_pair/60:,.1f} min) per pair")
        print(f"  ~{est_total:,.0f} s (~{est_total/3600:,.2f} hr) for all "
              f"{len(pair_results)} consecutive pairs")





"""
Loads all 10 pre-chunked genome segments (each a uint8-encoded .npy array,
A=0, C=1, G=2, T=3, N=4) and runs pure-Python Needleman-Wunsch global
alignment on each consecutive pair (0-1, 1-2, ..., 8-9), timing every
pair to establish a baseline performance metric across the full chunked
genome rather than just a single pair.
"""
