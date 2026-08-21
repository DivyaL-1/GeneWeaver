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
            yield Static("🧬 GENEWEAVER", classes="title")
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