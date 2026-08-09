from Bio import SeqIO

genome_file = "data/GCF_000001405.40_GRCh38.p14_genomic.fna"

CHUNK_SIZE = 1_000_000  # 1 million bases

total_records = 0
total_chunks = 0
total_bases = 0

for record in SeqIO.parse(genome_file, "fasta"):

    total_records += 1
    sequence_length = len(record.seq)
    total_bases += sequence_length

    chunk_count = (sequence_length + CHUNK_SIZE - 1) // CHUNK_SIZE
    total_chunks += chunk_count

    print(
        f"{record.id}: "
        f"{sequence_length:,} bases → "
        f"{chunk_count} chunks"
    )

print("\n========== SUMMARY ==========")
print(f"Total records : {total_records:,}")
print(f"Total bases   : {total_bases:,}")
print(f"Chunk size    : {CHUNK_SIZE:,} bases")
print(f"Total chunks  : {total_chunks:,}")
print("=============================")