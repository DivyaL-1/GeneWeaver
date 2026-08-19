from pathlib import Path
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Header, Footer, Static, ProgressBar


class GeneWeaverApp(App):

    TITLE = "GeneWeaver - Genome Processing Dashboard"

    def compose(self) -> ComposeResult:
        yield Header()

        with Container():
            yield Static("🧬 GeneWeaver", classes="title")
            yield Static("Genome Chunking Progress")

            yield ProgressBar(
                total=10,
                show_eta=False,
                id="chunk_progress"
            )

            yield Static("Status: Ready", id="status")
            yield Static("Chunks: 0 / 10", id="chunk_count")

        yield Footer()

    def on_mount(self) -> None:
        """Start processing when the app opens."""
        self.process_chunks()

    def process_chunks(self) -> None:
        """Process the available genome chunks."""

        chunk_dir = Path("data/chunks")
        chunk_files = sorted(chunk_dir.glob("chunk_*.npy"))

        progress_bar = self.query_one("#chunk_progress", ProgressBar)
        status = self.query_one("#status", Static)
        chunk_count = self.query_one("#chunk_count", Static)

        total_chunks = len(chunk_files)
        progress_bar.update(total=total_chunks)

        if total_chunks == 0:
            status.update("Status: No chunk files found")
            return

        status.update("Status: Processing genome chunks...")

        for index, chunk_file in enumerate(chunk_files, start=1):
            progress_bar.advance(1)
            chunk_count.update(f"Chunks: {index} / {total_chunks}")

        status.update("Status: Chunk processing complete")


if __name__ == "__main__":
    GeneWeaverApp().run()