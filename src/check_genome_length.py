from pathlib import Path
from Bio import SeqIO

genome_dir = Path("data")

total_bases = 0
sequence_count = 0

for file in genome_dir.rglob("*"):
    if file.is_file():
        try:
            for record in SeqIO.parse(file, "fasta"):
                length = len(record.seq)
                total_bases += length
                sequence_count += 1

                print(f"{record.id}: {length:,} bases")

        except Exception:
            pass

print("\n" + "=" * 50)
print("GENOME SUMMARY")
print("=" * 50)

print(f"Number of sequences : {sequence_count:,}")
print(f"Total genome length : {total_bases:,} bases")
print(f"Approx. size        : {total_bases / 1_000_000_000:.3f} billion bases")

chunk_size = 1_000_000
chunks = (total_bases + chunk_size - 1) // chunk_size

print(f"Chunk size           : {chunk_size:,} bases")
print(f"Chunks required      : {chunks:,}")