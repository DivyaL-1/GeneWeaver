import os
import time
import numpy as np



# SET THESE TWO VARIABLES TO YOUR CHUNK FILE LOCATION
CHUNK_DIR = "data//chunks"                                          # directory holding the chunk files
CHUNK_FILENAMES = [f"chunk_{i:06d}.npy" for i in range(1,11)]  # the 10 chunk filenames

SAMPLE_CHUNK_SIZE = 5000  #(None = full length)

INT_TO_BASE = {0: "A", 1: "C", 2: "G", 3: "T"}
UNKNOWN_BASE = "N"

def chunk_as_sequence(path, limit=None):
    """Load a .npy genome chunk and return it as a ACGTN string, auto-detects the array dtype"""
    array = np.load(path, allow_pickle=True)
    if limit is not None:
        array = array[:limit]

    k=array.dtype.kind
    if k in ("u", "i"):
        seq = "".join(INT_TO_BASE.get(int(x), UNKNOWN_BASE) for x in array)
    elif k == "S":
        seq = b"".join(array.tolist()).decode("ascii")
    elif k == "U":
        seq = "".join(array.tolist())
    elif k == "O":
        parts = []
        for x in array.tolist() if array.ndim else [array.item()]:
            parts.append(x.decode("ascii") if isinstance(x, bytes) else str(x))
        seq = "".join(parts)
    else:
        raise ValueError(f"Unrecognized chunk dtype '{array.dtype}' in {path}")
    
    return seq.upper()

def needleman(seq1, seq2, match=1, mismatch=-1, gap=-1):
    #Compute the needleman-wunsch alignment
    n, m = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1,n + 1):
        dp[i][0]=i*gap

    for j in range(1,m + 1):
        dp[0][j]=j*gap

    for i in range(1,n + 1):
        row=dp[i]
        prev_row=dp[i-1]
        s1=seq1[i-1]
        for j in range(1,m + 1):
            score=match if s1==seq2[j-1] else mismatch
            diag=prev_row[j-1]+score
            up=prev_row[j]+gap
            left=row[j-1]+gap
            row[j] = diag if diag >= up and diag >= left else (up if up >= left else left)

    return dp[n][m]


if __name__ == "__main__":
    path = [os.path.join(CHUNK_DIR, fname) for fname in CHUNK_FILENAMES]

    print(f"Loading {len(path)} chunks from {CHUNK_DIR}..." + (f", sampling first {SAMPLE_CHUNK_SIZE:,} bp of each" if SAMPLE_CHUNK_SIZE else " (full length)"))
    sequences = [chunk_as_sequence(p, SAMPLE_CHUNK_SIZE) for p in path]
    print(f"Loaded chunk lengths: {[len(s) for s in sequences]}\n")
    full_len = 1_000_000 
    pair_results = []

    print(f"{'Pair':>10} | {'Length':>8} | {'Time (s)':>10} | {'Cells/sec':>13} | {'Score':>8}")
    print("-" * 62)

    for i in range(len(sequences) - 1):
            seq1, seq2 = sequences[i], sequences[i + 1]
            start = time.perf_counter()
            score = needleman(seq1, seq2)
            elapsed = time.perf_counter() - start
    
            cells = len(seq1) * len(seq2)
            cells_per_sec = cells / elapsed if elapsed > 0 else float("inf")
            pair_results.append({"pair": f"{i}-{i+1}", "elapsed": elapsed,
                                  "cells_per_sec": cells_per_sec, "score": score})
    
            print(f"{i:>3}-{i+1:<6} | {len(seq1):>8,} | {elapsed:>10.4f} | "
                  f"{cells_per_sec:>13,.0f} | {score:>8}")
    
    total_time = sum(r["elapsed"] for r in pair_results)
    avg_time = total_time / len(pair_results)
    avg_throughput = sum(r["cells_per_sec"] for r in pair_results) / len(pair_results)

    print("-" * 62)
    print(f"Pairs aligned       : {len(pair_results)}")
    print(f"Total time          : {total_time:.4f} s")
    print(f"Average time/pair   : {avg_time:.4f} s")
    print(f"Average throughput  : {avg_throughput:,.0f} cells/sec")
    
    if SAMPLE_CHUNK_SIZE is not None and SAMPLE_CHUNK_SIZE < full_len:
        full_cells = full_len * full_len
        est_per_pair = full_cells / avg_throughput
        est_total = est_per_pair * len(pair_results)
        print(f"\nExtrapolated to full {full_len:,} bp chunks:")
        print(f"  ~{est_per_pair:,.0f} s (~{est_per_pair/60:,.1f} min) per pair")
        print(f"  ~{est_total:,.0f} s (~{est_total/3600:,.2f} hr) for all "f"{len(pair_results)} consecutive pairs")






"""
Loads all 10 pre-chunked genome segments (each a uint8-encoded .npy array,
A=0, C=1, G=2, T=3, N=4) and runs pure-Python Needleman-Wunsch global
alignment on each consecutive pair (0-1, 1-2, ..., 8-9), timing every
pair to establish a baseline performance metric across the full chunked
genome rather than just a single pair.
"""