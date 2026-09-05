import numpy as np
from pathlib import Path


def find_pam_sites(sequence, offset=0):
    """
    Find SpCas9 PAM sequences using the NGG pattern.

    Valid PAM examples:
        AGG
        CGG
        GGG
        TGG

    Parameters:
        sequence: DNA sequence string
        offset: starting position of the sequence in the original chunk

    Returns:
        List of PAM sites with their positions.
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
    Calculate a PAM-proximity score.

    A PAM closer to the candidate position receives a higher score.

    Score:
        100 = PAM is at candidate position
        0   = PAM is 50 or more bases away

    Returns:
        Dictionary containing:
        - score
        - score_normalized
        - nearest_pam
        - pam_position
        - distance
        - risk
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

    # Find the closest PAM to the candidate
    nearest_site = min(
        pam_sites,
        key=lambda site: abs(
            site["position"] - candidate_position
        )
    )

    distance = abs(
        nearest_site["position"] - candidate_position
    )

    # Calculate proximity score
    if distance <= max_distance:
        score = (
            (max_distance - distance) / max_distance
        ) * 100
    else:
        score = 0.0

    score = round(score, 2)
    score_normalized = round(score / 100, 4)

    # PAM-specific risk classification
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
    """
    Load a DNA sequence from a .npy genome chunk.
    """

    arr = np.load(path, allow_pickle=True)

    if arr.dtype.kind == "S":
        sequence = b"".join(arr.tolist()).decode("ascii")
    else:
        raise ValueError(
            f"Unsupported chunk dtype: {arr.dtype}"
        )

    return sequence.upper()


def get_first_valid_position(sequence):
    """
    Find the first valid DNA base (A/C/G/T).

    Returns:
        Position of the first valid DNA base,
        or None if no valid base exists.
    """

    return next(
        (
            i
            for i, base in enumerate(sequence)
            if base in "ACGT"
        ),
        None
    )


def analyze_all_chunks(
    chunk_directory,
    sample_length=5000
):
    """
    Find NGG PAM sites in all genome chunks.

    Only the first sample_length valid bases
    are analyzed from each chunk.

    Returns:
        Dictionary containing PAM analysis for each chunk.
    """

    chunk_directory = Path(chunk_directory)

    chunk_files = sorted(
        chunk_directory.glob("chunk_*.npy")
    )

    all_results = {}

    for chunk_path in chunk_files:

        sequence = load_chunk_as_sequence(chunk_path)

        start_position = get_first_valid_position(sequence)

        if start_position is None:
            all_results[chunk_path.name] = {
                "pam_count": 0,
                "pam_sites": []
            }
            continue

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
    Calculate PAM-proximity score for an off-target candidate.

    The function examines the region around the candidate
    position and searches for nearby NGG PAM sites.

    Parameters:
        chunk_path: Path to .npy genome chunk
        candidate_position: Candidate/off-target position
        sample_length: Number of bases to analyze
        max_distance: Maximum PAM distance considered

    Returns:
        PAM scoring dictionary.
    """

    sequence = load_chunk_as_sequence(chunk_path)

    start_position = get_first_valid_position(sequence)

    if start_position is None:
        return {
            "score": 0.0,
            "score_normalized": 0.0,
            "nearest_pam": None,
            "pam_position": None,
            "distance": None,
            "risk": "Low"
        }

    # Analyze a region around the candidate position
    sample_start = max(
        start_position,
        candidate_position - max_distance
    )

    sample_end = min(
        len(sequence),
        max(
            candidate_position + max_distance + 3,
            sample_start + sample_length
        )
    )

    sample = sequence[
        sample_start:sample_end
    ]

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

    chunk_path = (
        "data/chunks/chunk_000001.npy"
    )

    candidate_position = 10500

    result = score_candidate_pam(
        chunk_path,
        candidate_position
    )

    print()
    print("PAM CANDIDATE SCORING")
    print("-" * 40)

    print(
        f"Candidate Position: "
        f"{candidate_position}"
    )

    print(
        f"Nearest PAM       : "
        f"{result['nearest_pam']}"
    )

    print(
        f"PAM Position      : "
        f"{result['pam_position']}"
    )

    print(
        f"Distance          : "
        f"{result['distance']}"
    )

    print(
        f"PAM Score         : "
        f"{result['score']}"
    )

    print(
        f"Normalized Score  : "
        f"{result['score_normalized']}"
    )

    print(
        f"Risk Level        : "
        f"{result['risk']}"
    )

    print()