from pathlib import Path
import sys

from textual.app import App, ComposeResult
from textual import work
from textual.containers import Container
from textual.widgets import Header, Footer, Static, ProgressBar

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from GPUalgorithm import align_all_chunks_gpu
from numba import cuda


class GeneWeaverApp(App):

    CSS_PATH = "tui.tcss"

    TITLE = "GeneWeaver - Genome Processing Dashboard"

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()

        with Container(id="main-panel"):

            yield Static("🧬 GENEWEAVER", classes="title")

            # Alignment Dashboard
            yield Static(
                "ALIGNMENT DASHBOARD",
                classes="section-title"
            )

            yield Static(
                "Alignment Progress",
                id="alignment-progress-label"
            )

            yield Static(
                "Chunk Pair: 0 / 9",
                id="chunk-pair"
            )

            yield ProgressBar(
                total=9,
                show_eta=False,
                id="alignment_progress",
            )

            # GPU Status
            yield Static(
                "GPU STATUS",
                classes="section-title"
            )

            yield Static(
                "GPU: Checking...",
                id="gpu-status"
            )

            yield Static(
                "GPU Memory: --",
                id="gpu-memory"
            )

            yield Static(
                "GPU Utilization: --",
                id="gpu-utilization"
            )

            yield Static(
                "CUDA Status: Checking...",
                id="cuda-status"
            )

            # Results
            yield Static(
                "RESULTS",
                classes="section-title"
            )

            yield Static(
                "CPU Baseline",
                id="cpu-result-title"
            )

            yield Static(
                "Average Time: --",
                id="cpu-average-time"
            )

            yield Static(
                "Throughput: --",
                id="cpu-throughput"
            )

            yield Static(
                "GPU Result",
                id="gpu-result-title"
            )

            yield Static(
                "Average Time: --",
                id="gpu-average-time"
            )

            yield Static(
                "Throughput: --",
                id="gpu-throughput"
            )

            yield Static(
                "Speedup: --",
                id="gpu-speedup"
            )

            # Genome Chunking
            yield Static(
                "Genome Chunking Progress",
                classes="section-title"
            )

            yield ProgressBar(
                total=10,
                show_eta=False,
                id="chunk_progress",
            )

            yield Static(
                "Status: Ready",
                id="status"
            )

            yield Static(
                "Current file: None",
                id="current_file"
            )

            yield Static(
                "Chunks: 0 / 10",
                id="chunk_count"
            )

        yield Footer()

    def on_mount(self) -> None:

        self.update_gpu_status()

        self.chunk_files = sorted(
            Path("data/chunks").glob("chunk_*.npy")
        )

        self.current_chunk = 0

        progress_bar = self.query_one(
            "#chunk_progress",
            ProgressBar
        )

        progress_bar.update(
            total=len(self.chunk_files)
        )

        if not self.chunk_files:

            self.query_one(
                "#status",
                Static
            ).update(
                "Status: No chunk files found"
            )

            return

        self.query_one(
            "#status",
            Static
        ).update(
            "Status: Loading genome..."
        )

        self.set_timer(
            2,
            self.start_processing
        )

        self.run_gpu_alignment()

    def update_gpu_status(self) -> None:
        """Check whether CUDA is available."""

        cuda_available = cuda.is_available()

        if cuda_available:

            self.query_one(
                "#gpu-status",
                Static
            ).update(
                "GPU: Connected"
            )

            self.query_one(
                "#cuda-status",
                Static
            ).update(
                "CUDA Status: Available"
            )

            try:
                gpu = cuda.get_current_device()

                total_memory_gb = (
                    gpu.total_memory / (1024 ** 3)
                )

                self.query_one(
                    "#gpu-memory",
                    Static
                ).update(
                    f"GPU Memory: {total_memory_gb:.2f} GB"
                )

            except Exception:

                self.query_one(
                    "#gpu-memory",
                    Static
                ).update(
                    "GPU Memory: Available"
                )

            self.query_one(
                "#gpu-utilization",
                Static
            ).update(
                "GPU Utilization: Available"
            )

        else:

            self.query_one(
                "#gpu-status",
                Static
            ).update(
                "GPU: Not Connected"
            )

            self.query_one(
                "#cuda-status",
                Static
            ).update(
                "CUDA Status: Not Available"
            )

            self.query_one(
                "#gpu-memory",
                Static
            ).update(
                "GPU Memory: --"
            )

            self.query_one(
                "#gpu-utilization",
                Static
            ).update(
                "GPU Utilization: --"
            )

    @work(thread=True)
    def run_gpu_alignment(self) -> None:
        """Run GPU alignment without blocking the UI."""

        if not cuda.is_available():

            self.call_from_thread(
                self.query_one(
                    "#alignment-progress-label",
                    Static
                ).update,
                "Alignment: CUDA GPU not available on this system",
            )

            return

        def progress_callback(
            pair_index,
            total_pairs,
            done_diag,
            total_diag
        ):

            self.call_from_thread(
                self.update_alignment_progress,
                pair_index,
                total_pairs,
                done_diag,
                total_diag,
            )

        results = align_all_chunks_gpu(
            "data/chunks",
            [
                f"chunk_{i:06d}.npy"
                for i in range(1, 11)
            ],
            sample_size=5000,
            progress_callback=progress_callback,
        )

        self.call_from_thread(
            self.display_gpu_results,
            results,
        )

    def update_alignment_progress(
        self,
        pair_index,
        total_pairs,
        done_diag,
        total_diag,
    ) -> None:
        """Update alignment progress."""

        pair_number = pair_index + 1

        overall_progress = (
            pair_index +
            (done_diag / total_diag)
        ) / total_pairs

        progress_bar = self.query_one(
            "#alignment_progress",
            ProgressBar,
        )

        progress_bar.update(
            progress=overall_progress * total_pairs
        )

        self.query_one(
            "#chunk-pair",
            Static,
        ).update(
            f"Chunk Pair: {pair_number} / {total_pairs}"
        )

        self.query_one(
            "#alignment-progress-label",
            Static,
        ).update(
            f"Aligning pair {pair_number}/{total_pairs}: "
            f"{done_diag}/{total_diag} diagonals"
        )

    def display_gpu_results(
        self,
        results
    ) -> None:
        """Display GPU alignment results."""

        if not results:
            return

        total_seconds = sum(
            result["timings"]["total_s"]
            for result in results
        )

        average_seconds = (
            total_seconds / len(results)
        )

        self.query_one(
            "#gpu-average-time",
            Static,
        ).update(
            f"Average Time: {average_seconds * 1000:.3f} ms"
        )

        self.query_one(
            "#gpu-throughput",
            Static,
        ).update(
            f"Pairs aligned: {len(results)}"
        )

        self.query_one(
            "#alignment-progress-label",
            Static,
        ).update(
            "Alignment complete"
        )

        self.query_one(
            "#chunk-pair",
            Static,
        ).update(
            f"Chunk Pair: {len(results)} / {len(results)}"
        )

        self.query_one(
            "#alignment_progress",
            ProgressBar,
        ).update(
            progress=len(results)
        )

    def start_processing(self) -> None:
        """Begin processing genome chunks."""

        self.query_one(
            "#status",
            Static
        ).update(
            "Status: Processing genome chunks..."
        )

        self.set_interval(
            1,
            self.process_next_chunk
        )

    def process_next_chunk(self) -> None:
        """Update chunk progress."""

        if self.current_chunk >= len(self.chunk_files):

            self.query_one(
                "#status",
                Static
            ).update(
                "Status: Chunk processing complete"
            )

            self.query_one(
                "#current_file",
                Static
            ).update(
                "Current file: All chunks processed"
            )

            return

        chunk_file = self.chunk_files[
            self.current_chunk
        ]

        index = self.current_chunk + 1

        self.query_one(
            "#status",
            Static
        ).update(
            f"Status: Processing chunk "
            f"{index}/{len(self.chunk_files)}"
        )

        self.query_one(
            "#current_file",
            Static
        ).update(
            f"Current file: {chunk_file.name}"
        )

        self.query_one(
            "#chunk_count",
            Static
        ).update(
            f"Chunks: {index} / "
            f"{len(self.chunk_files)}"
        )

        self.query_one(
            "#chunk_progress",
            ProgressBar
        ).advance(1)

        self.current_chunk += 1


if __name__ == "__main__":
    GeneWeaverApp().run()