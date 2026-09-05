from Bio import SeqIO
import numpy as np
import os

genome_file = "data/GCF_000001405.40_GRCh38.p14_genomic.fna"
chunks_folder = "data/chunks"

CHUNK_SIZE = 1_000_000
MAX_CHUNKS = 30

os.makedirs(chunks_folder, exist_ok=True)

total_records = 0
total_bases = 0
total_chunks = 0

for record in SeqIO.parse(genome_file, "fasta"):

    total_records += 1
    sequence = record.seq
    sequence_length = len(sequence)
    total_bases += sequence_length

    for start in range(0, sequence_length, CHUNK_SIZE):

        if total_chunks >= MAX_CHUNKS:
            break

        end = min(start + CHUNK_SIZE, sequence_length)

        chunk = sequence[start:end]

        # Store each DNA base as 1 byte
        chunk_array = np.array(list(str(chunk)), dtype="S1")

        chunk_file = os.path.join(
            chunks_folder,
            f"chunk_{total_chunks + 1:06d}.npy"
        )

        np.save(chunk_file, chunk_array)

        total_chunks += 1

        print(
            f"Saved {chunk_file} | "
            f"Length: {len(chunk):,} bases"
        )

    if total_chunks >= MAX_CHUNKS:
        break

print("\n========== SUMMARY ==========")
print(f"Records read   : {total_records:,}")
print(f"Total bases    : {total_bases:,}")
print(f"Chunks created : {total_chunks:,}")
print(f"Chunk size     : {CHUNK_SIZE:,} bases")
print(f"Saved to       : {chunks_folder}")
print("=============================")