def calculate_severity(
    alignment_score,
    mismatch_count,
    pam_score
):
    """
    Calculate the final off-target severity score.

    Parameters:
        alignment_score: Alignment quality from 0 to 1.
        mismatch_count: Number of sequence mismatches.
        pam_score: Normalized PAM score from 0 to 1.

    Weights:
        Alignment = 50%
        Mismatch  = 20%
        PAM       = 30%

    Returns:
        Final severity score between 0 and 1.
    """

    # Validate scores
    alignment_score = max(
        0.0,
        min(1.0, float(alignment_score))
    )

    pam_score = max(
        0.0,
        min(1.0, float(pam_score))
    )

    mismatch_count = max(
        0,
        int(mismatch_count)
    )

    # Convert mismatch count into a score.
    #
    # 0 mismatches -> 1.0
    # 1 mismatch   -> 0.5
    # 2 mismatches -> 0.333
    # 3 mismatches -> 0.25
    mismatch_score = 1 / (
        1 + mismatch_count
    )

    # Biological scoring weights
    alignment_weight = 0.5
    mismatch_weight = 0.2
    pam_weight = 0.3

    final_score = (
        alignment_score * alignment_weight
        + mismatch_score * mismatch_weight
        + pam_score * pam_weight
    )

    return round(final_score, 4)


def classify_severity(score):
    """
    Convert numerical severity score into
    HIGH, MEDIUM, or LOW.
    """

    if score >= 0.75:
        return "HIGH"

    elif score >= 0.50:
        return "MEDIUM"

    else:
        return "LOW"


def add_severity_to_result(
    result,
    alignment_score,
    mismatch_count,
    pam_score
):
    """
    Add severity information to an off-target result.

    This connects the alignment information
    with the PAM score.
    """

    severity_score = calculate_severity(
        alignment_score=alignment_score,
        mismatch_count=mismatch_count,
        pam_score=pam_score
    )

    result["alignment_score"] = alignment_score
    result["mismatch_count"] = mismatch_count
    result["pam_score"] = pam_score
    result["severity_score"] = severity_score
    result["severity"] = classify_severity(
        severity_score
    )

    return result


def rank_off_targets(results):
    """
    Rank off-target results from highest
    to lowest severity.
    """

    return sorted(
        results,
        key=lambda result: result["severity_score"],
        reverse=True
    )


if __name__ == "__main__":

    # Example result
    alignment_score = 0.90
    mismatch_count = 1
    pam_score = 0.94

    severity_score = calculate_severity(
        alignment_score,
        mismatch_count,
        pam_score
    )

    severity = classify_severity(
        severity_score
    )

    print()
    print("OFF-TARGET SEVERITY")
    print("-" * 40)

    print(
        f"Alignment Score : {alignment_score}"
    )

    print(
        f"Mismatch Count  : {mismatch_count}"
    )

    print(
        f"PAM Score       : {pam_score}"
    )

    print(
        f"Severity Score  : {severity_score}"
    )

    print(
        f"Severity        : {severity}"
    )

    print()