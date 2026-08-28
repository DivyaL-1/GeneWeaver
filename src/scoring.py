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


if __name__ == "__main__":

    # -------------------------------------------------
    # 1. Load real genome chunk
    # -------------------------------------------------

    chunk_path = "data/chunks/chunk_000001.npy"

    sequence = load_chunk_as_sequence(chunk_path)

    print(f"Full chunk length: {len(sequence):,}")

    # -------------------------------------------------
    # 2. Find where real DNA begins
    # -------------------------------------------------

    start_position = next(
        (
            i
            for i, base in enumerate(sequence)
            if base in "ACGT"
        ),
        None
    )

    if start_position is None:
        print("No valid DNA bases found.")
        raise SystemExit

    # -------------------------------------------------
    # 3. Take 5,000 bases from the real DNA region
    # -------------------------------------------------

    sample_length = 5000

    sample = sequence[
        start_position:
        start_position + sample_length
    ]

    print(
        f"First valid DNA position: "
        f"{start_position}"
    )

    print(
        f"Sample sequence length: "
        f"{len(sample):,}"
    )

    # -------------------------------------------------
    # 4. Find PAM sites using actual chunk positions
    # -------------------------------------------------

    pam_sites = find_pam_sites(
        sample,
        offset=start_position
    )

    print(
        f"Total NGG PAM sites found: "
        f"{len(pam_sites):,}"
    )

    print("\nFirst 10 PAM sites:")

    for site in pam_sites[:10]:
        print(
            f"PAM: {site['pam']} | "
            f"Chunk position: {site['position']}"
        )

    # -------------------------------------------------
    # 5. Test PAM proximity scoring
    # -------------------------------------------------

    candidate_position = 10500

    result = calculate_pam_proximity_score(
        pam_sites,
        candidate_position
    )

    print("\nPAM Proximity Scoring Result")
    print("-" * 35)

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