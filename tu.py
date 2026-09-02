import numpy as np

if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack

import os
import time
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Header, Footer, Static, ProgressBar
from textual.worker import get_current_worker


import gpu as cuda_alignment
import firstAlgorithm as align_all_chunks


# >>> SET THESE TO YOUR CHUNK FILE LOCATION <<<
CHUNK_DIR = "data/chunks"
CHUNK_FILENAMES = [f"chunk_{i:06d}.npy" for i in range(1, 11)]

SAMPLE_SIZE = 5000


class GeneWeaverApp(App):

    CSS_PATH = "tui.tcss"

    TITLE = "GeneWeaver - Genome Processing Dashboard"

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()

        with Container(id="main-panel"):
            # GeneWeaver title
            yield Static("🧬 GENEWEAVER", classes="title")

            # Alignment Dashboard
            yield Static("ALIGNMENT DASHBOARD", classes="section-title")
            yield Static("Alignment Progress", id="alignment-progress-label")
            yield Static("Chunk Pair: 0 / 9", id="chunk-pair")

            yield ProgressBar(
                total=9,
                show_eta=False,
                id="alignment_progress",
            )

            # GPU Status
            yield Static("GPU STATUS", classes="section-title")
            yield Static("GPU: Not Connected", id="gpu-status")
            yield Static("GPU Count: --", id="gpu-count")
            yield Static("GPU Memory: --", id="gpu-memory")
            yield Static("GPU Utilization: --", id="gpu-utilization")
            yield Static("CUDA Status: Not Available", id="cuda-status")

            # Results
            yield Static("RESULTS", classes="section-title")

            yield Static("CPU Baseline", id="cpu-result-title")
            yield Static("Average Time: --", id="cpu-average-time")
            yield Static("Throughput: --", id="cpu-throughput")

            yield Static("GPU Result", id="gpu-result-title")
            yield Static("Average Time: --", id="gpu-average-time")
            yield Static("Throughput: --", id="gpu-throughput")
            yield Static("Speedup: --", id="gpu-speedup")


            yield Static("Genome Chunking Progress", classes="section-title")

            yield ProgressBar(
                total=10,
                show_eta=False,
                id="chunk_progress",
            )

            yield Static("Status: Ready", id="status")
            yield Static("Current file: None", id="current_file")
            yield Static("Chunks: 0 / 10", id="chunk_count")

        yield Footer()

    def on_mount(self) -> None:
        candidate_paths = [Path(CHUNK_DIR) / fname for fname in CHUNK_FILENAMES]
        missing = [p for p in candidate_paths if not p.exists()]

        progress_bar = self.query_one("#chunk_progress", ProgressBar)
        progress_bar.update(total=len(candidate_paths))

        if missing:
            shown = ", ".join(p.name for p in missing[:3])
            more = f" (+{len(missing) - 3} more)" if len(missing) > 3 else ""
            self.query_one("#status", Static).update(
                f"Status: Missing {len(missing)}/{len(candidate_paths)} chunk file(s) "
                f"in '{CHUNK_DIR}': {shown}{more}"
            )
            return

        self.chunk_files = candidate_paths
        self.query_one("#status", Static).update("Status: Loading genome...")
        self.run_pipeline_worker()

    def _set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(f"Status: {text}")

    def _set_current_file(self, text: str) -> None:
        self.query_one("#current_file", Static).update(f"Current file: {text}")

    def _set_chunk_progress(self, done: int, total: int) -> None:
        self.query_one("#chunk_progress", ProgressBar).update(progress=done)
        self.query_one("#chunk_count", Static).update(f"Chunks: {done} / {total}")

    def _set_alignment_label(self, text: str) -> None:
        self.query_one("#alignment-progress-label", Static).update(text)

    def _set_alignment_progress(self, done: int, total: int) -> None:
        self.query_one("#alignment_progress", ProgressBar).update(total=total, progress=done)
        self.query_one("#chunk-pair", Static).update(f"Chunk Pair: {done} / {total}")

    def _set_gpu_status_widgets(self, info: dict) -> None:
        if info["simulator"]:
            self.query_one("#gpu-status", Static).update("GPU: Simulator (no hardware detected)")
            self.query_one("#cuda-status", Static).update("CUDA Status: Simulator mode")
        elif info["available"]:
            name = info["device_name"] or "GPU"
            self.query_one("#gpu-status", Static).update(f"GPU: Connected ({name})")
            self.query_one("#cuda-status", Static).update("CUDA Status: Available")
        else:
            self.query_one("#gpu-status", Static).update("GPU: Not Connected")
            self.query_one("#cuda-status", Static).update("CUDA Status: Not Available")

        n_gpus = cuda_alignment.detect_gpu_count()
        if info["simulator"]:
            self.query_one("#gpu-count", Static).update(f"GPU Count: {n_gpus} (simulated)")
        else:
            label = "GPU" if n_gpus == 1 else "GPUs"
            self.query_one("#gpu-count", Static).update(f"GPU Count: {n_gpus} {label} detected")

        if info["memory_free_gb"] is not None:
            self.query_one("#gpu-memory", Static).update(
                f"GPU Memory: {info['memory_free_gb']:.2f} / {info['memory_total_gb']:.2f} GB free"
            )
        else:
            self.query_one("#gpu-memory", Static).update("GPU Memory: N/A")

        if info["utilization_pct"] is not None:
            self.query_one("#gpu-utilization", Static).update(
                f"GPU Utilization: {info['utilization_pct']:.0f}%"
            )
        else:
            self.query_one("#gpu-utilization", Static).update("GPU Utilization: N/A")

    def _set_cpu_results(self, avg_time_s: float, avg_throughput: float) -> None:
        self.query_one("#cpu-average-time", Static).update(f"Average Time: {avg_time_s*1000:.2f} ms")
        self.query_one("#cpu-throughput", Static).update(f"Throughput: {avg_throughput:,.0f} cells/sec")

    def _set_gpu_results(self, avg_time_s: float, avg_throughput: float, speedup: float) -> None:
        self.query_one("#gpu-average-time", Static).update(f"Average Time: {avg_time_s*1000:.2f} ms")
        self.query_one("#gpu-throughput", Static).update(f"Throughput: {avg_throughput:,.0f} cells/sec")
        self.query_one("#gpu-speedup", Static).update(f"Speedup: {speedup:.1f}x")


    @work(thread=True)
    def run_pipeline_worker(self) -> None:
        worker = get_current_worker()


        gpu_info = cuda_alignment.gpu_status_info()
        self.call_from_thread(self._set_gpu_status_widgets, gpu_info)


        self.call_from_thread(self._set_status, "Loading genome...")
        sequences = []
        for idx, path in enumerate(self.chunk_files, start=1):
            if worker.is_cancelled:
                return
            seq = cuda_alignment.load_chunk_as_sequence(str(path), SAMPLE_SIZE)
            sequences.append(seq)

            self.call_from_thread(self._set_current_file, path.name)
            self.call_from_thread(self._set_chunk_progress, idx, len(self.chunk_files))
            self.call_from_thread(
                self._set_status, f"Processing chunk {idx}/{len(self.chunk_files)}"
            )

        if worker.is_cancelled:
            return


        self.call_from_thread(self._set_status, "Running CPU baseline...")
        self.call_from_thread(self._set_alignment_label, "CPU Alignment Progress")
        self.call_from_thread(self._set_alignment_progress, 0, len(sequences) - 1)

        def cpu_progress_cb(pair_index, n_pairs):
            self.call_from_thread(self._set_alignment_progress, pair_index + 1, n_pairs)

        cpu_results = align_all_chunks.align_sequence_pairs_cpu(
            sequences, progress_callback=cpu_progress_cb
        )
        cpu_avg_time = sum(r["elapsed_s"] for r in cpu_results) / len(cpu_results)
        cpu_avg_throughput = sum(r["cells_per_sec"] for r in cpu_results) / len(cpu_results)
        self.call_from_thread(self._set_cpu_results, cpu_avg_time, cpu_avg_throughput)

        if worker.is_cancelled:
            return

        self.call_from_thread(self._set_status, "Running GPU benchmark...")
        self.call_from_thread(self._set_alignment_label, "GPU Alignment Progress")
        self.call_from_thread(self._set_alignment_progress, 0, len(sequences) - 1)

        def gpu_progress_cb(pair_index, n_pairs, done_diag, total_diag):
            if done_diag == total_diag:
                self.call_from_thread(self._set_alignment_progress, pair_index + 1, n_pairs)

        # --------------------------------------------------------
        # DASK MULTI-GPU ALIGNMENT
        # --------------------------------------------------------

        self.call_from_thread(
            self._set_status,
            "Running Dask multi-GPU alignment..."
        )

        try:
            dask_results, n_gpus = (
                cuda_alignment.align_all_chunks_gpu_dask(
                    CHUNK_DIR,
                    CHUNK_FILENAMES,
                    SAMPLE_SIZE,
                )
            )
        except Exception as exc:
            self.call_from_thread(
                self._set_status,
                f"Dask/GPU error: {exc}"
            )
            return

        if worker.is_cancelled:
            return

        if not dask_results:
            self.call_from_thread(
                self._set_status,
                "No Dask alignment results"
            )
            return

        # Update alignment progress after Dask completes.
        self.call_from_thread(
            self._set_alignment_progress,
            len(dask_results),
            len(dask_results),
        )

        # --------------------------------------------------------
        # Calculate Dask GPU results
        # --------------------------------------------------------

        gpu_avg_time = (
            sum(r["wall_s"] for r in dask_results)
            / len(dask_results)
        )

        gpu_avg_throughput = (
            sum(
                (r["n"] * r["m"]) / r["wall_s"]
                for r in dask_results
                if r["wall_s"] > 0
            )
            / len(dask_results)
        )

        speedup = (
            cpu_avg_time / gpu_avg_time
            if gpu_avg_time > 0
            else float("inf")
        )

        self.call_from_thread(
            self._set_gpu_results,
            gpu_avg_time,
            gpu_avg_throughput,
            speedup,
        )

        # --------------------------------------------------------
        # Show Dask workload distribution
        # --------------------------------------------------------

        per_gpu_counts = {}

        for result in dask_results:
            gpu_id = result.get("gpu_id", 0)
            per_gpu_counts[gpu_id] = (
                per_gpu_counts.get(gpu_id, 0) + 1
            )

        distribution = ", ".join(
            f"GPU {gpu_id}: {count} pair(s)"
            for gpu_id, count in sorted(per_gpu_counts.items())
        )

        self.call_from_thread(
            self._set_status,
            f"Dask complete - {n_gpus} GPU(s) | {distribution}"
        )

        self.call_from_thread(self._set_current_file, "All chunks processed")


if __name__ == "__main__":
    GeneWeaverApp().run()
