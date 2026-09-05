from severity_scoring import (
    calculate_severity,
    classify_severity,
    rank_off_targets,
)


def test_severity_calculation():

    score = calculate_severity(
        alignment_score=0.9,
        mismatch_count=1,
        pam_score=0.9,
    )

    print("Severity score:", score)

    assert 0 <= score <= 1


def test_severity_classification():

    assert classify_severity(0.80) == "HIGH"

    assert classify_severity(0.60) == "MEDIUM"

    assert classify_severity(0.30) == "LOW"


def test_ranking():

    results = [
        {
            "name": "Off-target A",
            "severity_score": 0.60
        },
        {
            "name": "Off-target B",
            "severity_score": 0.90
        },
        {
            "name": "Off-target C",
            "severity_score": 0.40
        },
    ]

    ranked = rank_off_targets(results)

    assert ranked[0]["name"] == "Off-target B"

    assert ranked[1]["name"] == "Off-target A"

    assert ranked[2]["name"] == "Off-target C"


def test_severity_with_good_pam():

    score = calculate_severity(
        alignment_score=0.90,
        mismatch_count=1,
        pam_score=0.94,
    )

    assert score == 0.832

    assert classify_severity(score) == "HIGH"


if __name__ == "__main__":

    test_severity_calculation()
    test_severity_classification()
    test_ranking()
    test_severity_with_good_pam()

    print("All severity scoring tests passed!")