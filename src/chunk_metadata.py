import json
import os
import glob
import numpy as np

CHUNK_DIR = "data/chunks"
METADATA_FILE = os.path.join(CHUNK_DIR, "metadata.json")


def create_metadata():
    chunk_files = sorted(glob.glob(os.path.join(CHUNK_DIR, "chunk_*.npy")))

    metadata = {
        "total_chunks": len(chunk_files),
        "chunks": []
    }

    for index, file_path in enumerate(chunk_files, start=1):
        array = np.load(file_path, allow_pickle=False)

        chunk_info = {
            "chunk_number": index,
            "file": os.path.basename(file_path),
            "length": int(array.shape[0]),
            "dtype": str(array.dtype)
        }

        metadata["chunks"].append(chunk_info)

    with open(METADATA_FILE, "w") as file:
        json.dump(metadata, file, indent=4)

    print(f"Metadata saved to: {METADATA_FILE}")
    print(f"Total chunks: {len(chunk_files)}")


if __name__ == "__main__":
    create_metadata()