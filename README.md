# GeneWeaver

## Week 1

### Tasks
1. Develop a Pure Python algorithm to establish a CPU baseline performance metric.
2. Obtain and process a genome dataset.
3. Form manageable chunks from the genome data.
4. Store genome chunks as NumPy arrays.
5. Generate metadata for the chunks.
6. Validate chunk shape, dtype, length, and DNA bases.

### Current Progress

- Genome parsed using BioPython.
- Genome divided into 1,000,000-base chunks.
- 10 development chunks created.
- Chunks stored as NumPy `.npy` arrays.
- Metadata generated for all chunks.
- Validation tests implemented and passed.
- CPU baseline implemented separately using the chunked genome data.

### Dataset

The original genome dataset is approximately 3.2 GB and is not included in the repository.

For development and testing, 10 NumPy chunks are included, each containing 1,000,000 bases.

### Testing

All 10 chunks were successfully validated for:
- Correct shape
- Correct dtype
- Correct length
- Valid DNA bases
- Metadata availability