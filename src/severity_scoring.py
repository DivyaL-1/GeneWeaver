def calculate_severity(alignment_score, mismatch_count, pam_score):
    """
    Calculate final off-target severity score.

    pam_score must be normalized between 0 and 1.
    """

    # Placeholder weights
    alignment_weight = 0.5
    mismatch_weight = 0.2
    pam_weight = 0.3

    # Convert mismatch count into a score.
    mismatch_score = 1 / (1 + mismatch_count)

    final_score = (
        alignment_score * alignment_weight
        + mismatch_score * mismatch_weight
        + pam_score * pam_weight
    )

    return final_score


def classify_severity(score):
    """
    Convert numerical score into severity level.
    """

    if score >= 0.75:
        return "HIGH"
    elif score >= 0.50:
        return "MEDIUM"
    else:
        return "LOW"


def rank_off_targets(results):
    """
    Rank off-target results from highest to lowest severity.
    """

    return sorted(
        results,
        key=lambda result: result["severity_score"],
        reverse=True
    )
