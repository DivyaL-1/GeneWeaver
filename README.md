# GeneWeaver

GeneWeaver is a project focused on high-throughput genome sequence processing
and sequence alignment using parallel/GPU-based approaches.

## Current Work – Data Pipeline

The current task is to prepare genome data for further processing by the
GeneWeaver pipeline.

### Input Dataset

The current dataset is the NCBI GRCh38.p14 human reference genome:

`GCF_000001405.40_GRCh38.p14_genomic.fna`

The genome file is approximately 3.26 GB.

The genome file is kept locally and is not committed to Git because of its
large size.

## Work Completed

- Created the GeneWeaver project structure.
- Installed BioPython.
- Used BioPython `SeqIO` to parse the FASTA genome file.
- Successfully processed the NCBI genome dataset.
- Identified 705 sequence records.
- Calculated a total of 3,298,430,636 DNA bases.
- Selected a chunk size of 1,000,000 bases.
- Calculated 3,857 manageable chunks.
- Created the initial genome parsing and chunk calculation pipeline.

## Current Pipeline


NCBI GRCh38.p14 FASTA
        ↓
     BioPython
        ↓
   FASTA parsing
        ↓
  Sequence records
        ↓
  1,000,000-base chunks
        ↓
  Manageable genome data