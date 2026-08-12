import os
import glob
import numpy as np

CHUNK_DIR = "data/chunks"
EXPECTED_CHUNK_SIZE = 1_000_000


def test_chunk_files_exist():
    files = sorted(glob.glob(os.path.join(CHUNK_DIR, "chunk_*.npy")))

    assert len(files) == 10, f"Expected 10 chunks, found {len(files)}"

    print(f"PASS: Found {len(files)} chunk files")


def test_chunk_shapes_and_dtype():
    files = sorted(glob.glob(os.path.join(CHUNK_DIR, "chunk_*.npy")))

    for file_path in files:
        array = np.load(file_path, allow_pickle=False)

        assert array.ndim == 1, f"{file_path} is not a 1D array"
        assert array.dtype.kind == "S", f"{file_path} has unexpected dtype {array.dtype}"
        assert len(array) == EXPECTED_CHUNK_SIZE, (
            f"{file_path} has {len(array)} bases"
        )

    print("PASS: All chunks have correct shape, dtype, and length")


def test_chunk_contents():
    files = sorted(glob.glob(os.path.join(CHUNK_DIR, "chunk_*.npy")))

    valid_bases = {b"A", b"C", b"G", b"T", b"N", b"a", b"c", b"g", b"t", b"n"}

    for file_path in files:
        array = np.load(file_path, allow_pickle=False)

        invalid = set(array.tolist()) - valid_bases

        assert not invalid, (
            f"{file_path} contains invalid bases: {invalid}"
        )

    print("PASS: All chunks contain only valid DNA bases")


def test_metadata():
    metadata_path = os.path.join(CHUNK_DIR, "metadata.json")

    assert os.path.exists(metadata_path), "metadata.json is missing"

    print("PASS: metadata.json exists")


if __name__ == "__main__":
    test_chunk_files_exist()
    test_chunk_shapes_and_dtype()
    test_chunk_contents()
    test_metadata()

    print("\n========== ALL TESTS PASSED ==========")