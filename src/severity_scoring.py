import math

BASES = "ACGT"

TRANSITIONS = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}

TRANSITION_PENALTY = 0.6
TRANSVERSION_PENALTY = 1.0
UNKNOWN_BASE_PENALTY = 0.8

SEED_DECAY = 6.0

PAM_CLASS_WEIGHT = {"NGG": 1.00, "NAG": 0.25, "NGA": 0.10}
NON_CANONICAL_PAM_WEIGHT = 0.02


def substitution_penalty(guide_base, site_base):
    if guide_base == site_base:
        return 0.0
    if guide_base not in BASES or site_base not in BASES:
        return UNKNOWN_BASE_PENALTY
    if (guide_base, site_base) in TRANSITIONS:
        return TRANSITION_PENALTY
    return TRANSVERSION_PENALTY


def position_weight(index, guide_length, decay=SEED_DECAY):
    distance_to_pam = guide_length - 1 - index
    return math.exp(-distance_to_pam / decay)


def pam_class(pam):
    if not pam or len(pam) < 3:
        return None
    key = "N" + pam[1:3].upper()
    return key if key in PAM_CLASS_WEIGHT else None


def pam_score(pam, mismatches, guide_length, decay=SEED_DECAY):
    weight = PAM_CLASS_WEIGHT.get(pam_class(pam), NON_CANONICAL_PAM_WEIGHT)
    if guide_length <= 0:
        return weight

    disruption = 0.0
    for mm in mismatches:
        disruption += (position_weight(mm["index"], guide_length, decay)
                       * substitution_penalty(mm["guide_base"], mm["site_base"]))

    worst = sum(position_weight(i, guide_length, decay) for i in range(guide_length))
    activity = 1.0 - min(1.0, disruption / worst) if worst > 0 else 1.0
    return weight * activity


def normalize_alignment_score(raw_score, n, m, match=1, mismatch=-1, gap=-2):
    shorter, delta = min(n, m), abs(n - m)
    best = match * shorter + gap * delta
    worst = mismatch * shorter + gap * delta
    if best == worst:
        return 1.0
    return max(0.0, min(1.0, (raw_score - worst) / (best - worst)))


def calculate_severity(alignment_score, mismatch_count, pam_score):
    alignment_weight = 0.5
    mismatch_weight = 0.2
    pam_weight = 0.3

    mismatch_score = 1 / (1 + mismatch_count)

    final_score = (
        alignment_score * alignment_weight
        + mismatch_score * mismatch_weight
        + pam_score * pam_weight
    )

    return final_score


def classify_severity(score):
    if score >= 0.75:
        return "HIGH"
    elif score >= 0.50:
        return "MEDIUM"
    else:
        return "LOW"


def rank_off_targets(results):
    return sorted(
        results,
        key=lambda result: (
            -result["severity_score"],
            result.get("mismatch_count", 0),
            result.get("chunk", ""),
            result.get("start", 0),
        ),
    )


def score_site(site, decay=SEED_DECAY):
    guide_length = len(site["guide"])
    site["pam_class"] = pam_class(site["pam"]) or "non-canonical"
    site["pam_score"] = pam_score(site["pam"], site["mismatches"], guide_length, decay)
    site["alignment_score"] = normalize_alignment_score(
        site["raw_alignment_score"], guide_length, guide_length
    )
    site["severity_score"] = calculate_severity(
        site["alignment_score"], site["mismatch_count"], site["pam_score"]
    )
    site["severity"] = classify_severity(site["severity_score"])
    return site