from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Header, Footer, Static, ProgressBar


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

            # Existing genome chunking section
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
        self.chunk_files = sorted(Path("data/chunks").glob("chunk_*.npy"))
        self.current_chunk = 0

        progress_bar = self.query_one("#chunk_progress", ProgressBar)
        progress_bar.update(total=len(self.chunk_files))

        if not self.chunk_files:
            self.query_one("#status", Static).update(
                "Status: No chunk files found"
            )
            return

        self.query_one("#status", Static).update(
            "Status: Loading genome..."
        )

        # Show loading status briefly before processing.
        self.set_timer(2, self.start_processing)

    def start_processing(self) -> None:
        """Begin processing the genome chunks."""
        self.query_one("#status", Static).update(
            "Status: Processing genome chunks..."
        )

        self.set_interval(1, self.process_next_chunk)

    def process_next_chunk(self) -> None:
        """Update progress for the next genome chunk."""

        if self.current_chunk >= len(self.chunk_files):
            self.query_one("#status", Static).update(
                "Status: Chunk processing complete"
            )
            self.query_one("#current_file", Static).update(
                "Current file: All chunks processed"
            )
            return

        chunk_file = self.chunk_files[self.current_chunk]
        index = self.current_chunk + 1

        self.query_one("#status", Static).update(
            f"Status: Processing chunk {index}/{len(self.chunk_files)}"
        )

        self.query_one("#current_file", Static).update(
            f"Current file: {chunk_file.name}"
        )

        self.query_one("#chunk_count", Static).update(
            f"Chunks: {index} / {len(self.chunk_files)}"
        )

        self.query_one("#chunk_progress", ProgressBar).advance(1)

        self.current_chunk += 1


if __name__ == "__main__":
    GeneWeaverApp().run()
