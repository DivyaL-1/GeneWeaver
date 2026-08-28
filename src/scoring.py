import numpy as np


def find_pam_sites(sequence, offset=0):
    """
    Find SpCas9 PAM sequences with the NGG pattern.

    Valid examples:
    AGG, CGG, GGG, TGG

    offset converts sample positions into actual chunk positions.
    """

    sequence = sequence.upper()
    pam_sites = []

    for i in range(len(sequence) - 2):
        pam = sequence[i:i + 3]

        if pam[1:] == "GG":
            pam_sites.append({
                "position": i + offset,
                "pam": pam
            })

    return pam_sites


def calculate_pam_proximity_score(
    pam_sites,
    candidate_position,
    max_distance=50
):
    """
    Calculate PAM proximity score.

    Parameters:
        pam_sites: PAM sites with actual chunk positions
        candidate_position: off-target/candidate position
        max_distance: maximum distance considered for scoring

    Returns:
        Dictionary containing PAM score and normalized score.
    """

    if not pam_sites:
        return {
            "score": 0.0,
            "score_normalized": 0.0,
            "nearest_pam": None,
            "pam_position": None,
            "distance": None,
            "risk": "Low"
        }

    nearest_site = min(
        pam_sites,
        key=lambda site: abs(
            site["position"] - candidate_position
        )
    )

    distance = abs(
        nearest_site["position"] - candidate_position
    )

    if distance <= max_distance:
        score = (
            (max_distance - distance) / max_distance
        ) * 100
    else:
        score = 0.0

    score = round(score, 2)
    score_normalized = round(score / 100, 4)

    if score >= 70:
        risk = "High"
    elif score >= 40:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "score": score,
        "score_normalized": score_normalized,
        "nearest_pam": nearest_site["pam"],
        "pam_position": nearest_site["position"],
        "distance": distance,
        "risk": risk
    }


def load_chunk_as_sequence(path):
    """Load a DNA sequence from a .npy chunk file."""

    arr = np.load(path, allow_pickle=True)

    if arr.dtype.kind == "S":
        sequence = b"".join(arr.tolist()).decode("ascii")
    else:
        raise ValueError(
            f"Unsupported chunk dtype: {arr.dtype}"
        )

    return sequence.upper()

from pathlib import Path


def analyze_all_chunks(chunk_directory, sample_length=5000):
    """
    Find NGG PAM sites in all genome chunk files.

    Returns a dictionary where each chunk name maps to
    its PAM analysis results.
    """

    chunk_directory = Path(chunk_directory)

    chunk_files = sorted(
        chunk_directory.glob("chunk_*.npy")
    )

    all_results = {}

    for chunk_path in chunk_files:
        sequence = load_chunk_as_sequence(chunk_path)

        # Find the first valid DNA base
        start_position = next(
            (
                i
                for i, base in enumerate(sequence)
                if base in "ACGT"
            ),
            None
        )

        if start_position is None:
            all_results[chunk_path.name] = {
                "pam_count": 0,
                "pam_sites": []
            }
            continue

        # Analyze a sample from the first valid DNA region
        sample = sequence[
            start_position:
            start_position + sample_length
        ]

        pam_sites = find_pam_sites(
            sample,
            offset=start_position
        )

        all_results[chunk_path.name] = {
            "pam_count": len(pam_sites),
            "pam_sites": pam_sites
        }

    return all_results
def score_candidate_pam(
    chunk_path,
    candidate_position,
    sample_length=5000,
    max_distance=50
):
    """
    Score the PAM proximity for an off-target candidate.

    Parameters:
        chunk_path: Path to the .npy chunk file.
        candidate_position: Position within the chunk.
        sample_length: Number of bases to analyze.
        max_distance: Maximum PAM distance used for scoring.

    Returns:
        Dictionary containing PAM proximity scoring results.
    """

    sequence = load_chunk_as_sequence(chunk_path)

    start_position = next(
        (
            i
            for i, base in enumerate(sequence)
            if base in "ACGT"
        ),
        None
    )

    if start_position is None:
        return {
            "score": 0.0,
            "score_normalized": 0.0,
            "nearest_pam": None,
            "pam_position": None,
            "distance": None,
            "risk": "Low"
        }

    # Take a sample beginning from the candidate position.
    # This avoids always analyzing only the first 5,000 valid bases.
    sample_start = max(start_position, candidate_position - max_distance)
    sample_end = min(
        len(sequence),
        max(
            candidate_position + max_distance + 3,
            sample_start + sample_length
        )
    )

    sample = sequence[sample_start:sample_end]

    pam_sites = find_pam_sites(
        sample,
        offset=sample_start
    )

    return calculate_pam_proximity_score(
        pam_sites,
        candidate_position,
        max_distance=max_distance
    )

if __name__ == "__main__":

    chunk_path = "data/chunks/chunk_000001.npy"
    candidate_position = 10500

    result = score_candidate_pam(
        chunk_path,
        candidate_position
    )

    print("PAM CANDIDATE SCORING")
    print("-" * 35)
    print(f"Candidate Position: {candidate_position}")
    print(f"Nearest PAM       : {result['nearest_pam']}")
    print(f"PAM Position      : {result['pam_position']}")
    print(f"Distance          : {result['distance']}")
    print(f"PAM Score         : {result['score']}")
    print(f"Normalized Score  : {result['score_normalized']}")
    print(f"Risk Level        : {result['risk']}")