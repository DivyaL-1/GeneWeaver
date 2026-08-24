# GeneWeaver

GeneWeaver is a GPU-accelerated genomic sequence alignment project. It processes genomic data into manageable chunks and uses a CUDA-based implementation of the Needleman–Wunsch algorithm to perform sequence alignment.

The project includes a Textual terminal user interface that displays genome chunking progress, alignment progress, GPU status, and CPU vs GPU benchmark results.

## Features

* Parse and process genomic data
* Split genome data into manageable `.npy` chunks
* Display genome chunking progress
* Perform GPU-accelerated sequence alignment using CUDA
* Track alignment progress for each chunk pair
* Detect CUDA and GPU availability
* Display GPU memory and utilization information
* Compare CPU and GPU performance
* Calculate GPU speedup
* Display results in a Textual terminal dashboard

## Project Structure

```text
GeneWeaver/
├── data/
│   ├── chunks/
│   └── GCF_00001405.40_GRCh38.p14_genomic...
│
├── src/
│   ├── __pycache__/
│   ├── chunk_metadata.py
│   ├── firstAlgorithm.py
│   ├── genome_parser.py
│   ├── GPUalgorithm.py
│   ├── test_chunks.py
│   ├── tui.py
│   └── tui.tcss
│
├── .gitignore
└── README.md
```

## Technologies Used

* Python
* NumPy
* Numba CUDA
* Textual
* NVIDIA CUDA GPU
* Needleman–Wunsch sequence alignment algorithm

## GPU Alignment

The GPU implementation uses a diagonal parallelization approach for the Needleman–Wunsch dynamic programming algorithm.

The main steps are:

1. Load genomic chunks
2. Convert DNA bases into integer codes
3. Transfer sequences and the dynamic programming matrix to the GPU
4. Process the alignment matrix diagonal by diagonal
5. Align consecutive chunk pairs
6. Track alignment progress
7. Display timing and performance results

The project currently aligns 10 chunks, producing 9 consecutive chunk pairs:

```text
0-1
1-2
2-3
3-4
4-5
5-6
6-7
7-8
8-9
```

## Running the Textual Dashboard

From the project root:

```powershell
python src\tui.py
```

The Textual dashboard displays:

```text
GENEWEAVER

ALIGNMENT DASHBOARD
GPU Alignment Progress
Chunk Pair: 9 / 9
Progress: 100%

GPU STATUS
GPU: Connected
GPU Memory
GPU Utilization
CUDA Status

RESULTS
CPU Baseline
GPU Result
Speedup

Genome Chunking Progress
Status
Current File
Chunks: 10 / 10
```

## GPU Requirements

To run the GPU alignment, the system requires:

* An NVIDIA GPU with CUDA support
* CUDA-compatible drivers
* Numba
* NumPy

Check whether CUDA is available:

```powershell
nvidia-smi
```

The Python code also checks CUDA using:

```python
cuda.is_available()
```

If no CUDA-compatible GPU is available, the application displays:

```text
GPU: Not Connected
CUDA Status: Not Available
```

The dashboard can still run, but the GPU alignment itself requires CUDA hardware.

## Example Benchmark Result

The project was successfully tested on an:

```text
NVIDIA GeForce RTX 4060 Laptop GPU
```

Example results:

```text
CPU Average Time: 19853.55 ms
CPU Throughput:   5,038,904 cells/sec

GPU Average Time: 1362.30 ms
GPU Throughput:   75,334,582 cells/sec

GPU Speedup:      14.6x
```

This shows that the GPU implementation significantly improves sequence alignment performance compared with the CPU baseline.

## Current Status

* Genome parsing
* Genome chunking
* Chunk metadata generation
* Textual terminal interface
* Genome chunking progress bar
* Alignment progress tracking
* CUDA GPU detection
* GPU status display
* CPU baseline benchmark
* GPU alignment benchmark
* CPU vs GPU comparison
* Speedup calculation

## Team

Developed collaboratively as part of the GeneWeaver project.
