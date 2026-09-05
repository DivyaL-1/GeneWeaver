import numpy as np

import GPUalgorithm as gpu
import severity_scoring

CODE_A, CODE_C, CODE_G, CODE_T, CODE_N = 0, 1, 2, 3, 4
INT_TO_BASE = {0: "A", 1: "C", 2: "G", 3: "T", 4: "N"}

DEFAULT_GUIDE_LENGTH = 20
PAM_LENGTH = 3
MAX_HITS_PER_CHUNK = 500

_COMPLEMENT = np.array([CODE_T, CODE_G, CODE_C, CODE_A, CODE_N], dtype=np.int8)


def codes_to_str(codes):
    return "".join(INT_TO_BASE.get(int(c), "N") for c in codes)


def reverse_complement_codes(codes):
    codes = np.asarray(codes, dtype=np.int8)
    return _COMPLEMENT[np.clip(codes, 0, CODE_N)][::-1].copy()


def _scan_strand(codes, guide_codes, max_mismatches, strand, chunk_label, total_len):
    guide_len = len(guide_codes)
    window = guide_len + PAM_LENGTH
    n = len(codes)
    limit = n - window + 1
    if limit <= 0:
        return []

    mismatches = np.zeros(limit, dtype=np.int16)
    for k in range(guide_len):
        mismatches += (codes[k:k + limit] != guide_codes[k])

    p1 = codes[guide_len + 1: guide_len + 1 + limit]
    p2 = codes[guide_len + 2: guide_len + 2 + limit]
    keep = (mismatches <= max_mismatches) & (p2 == CODE_G) & ((p1 == CODE_G) | (p1 == CODE_A))

    hits = np.flatnonzero(keep)
    if hits.size > MAX_HITS_PER_CHUNK:
        hits = hits[np.argsort(mismatches[hits], kind="stable")[:MAX_HITS_PER_CHUNK]]

    results = []
    for i in hits:
        i = int(i)
        site_codes = codes[i:i + guide_len]
        pam_codes = codes[i + guide_len:i + window]

        diff = np.flatnonzero(site_codes != guide_codes)
        mm_list = [{
            "index": int(d),
            "guide_base": INT_TO_BASE.get(int(guide_codes[d]), "N"),
            "site_base": INT_TO_BASE.get(int(site_codes[d]), "N"),
            "distance_to_pam": guide_len - 1 - int(d),
        } for d in diff]

        start = i if strand == "+" else total_len - i - guide_len

        n_mm = len(mm_list)
        results.append({
            "chunk": chunk_label,
            "strand": strand,
            "start": start,
            "guide": codes_to_str(guide_codes),
            "site": codes_to_str(site_codes),
            "pam": codes_to_str(pam_codes),
            "mismatches": mm_list,
            "mismatch_count": n_mm,
            "raw_alignment_score": (guide_len - n_mm) - n_mm,
        })
    return results


def scan_chunk(codes, guide_codes, max_mismatches=4, chunk_label="", both_strands=True):
    codes = np.asarray(codes, dtype=np.int8)
    guide_codes = np.asarray(guide_codes, dtype=np.int8)
    total = len(codes)

    hits = _scan_strand(codes, guide_codes, max_mismatches, "+", chunk_label, total)
    if both_strands:
        hits += _scan_strand(reverse_complement_codes(codes), guide_codes,
                             max_mismatches, "-", chunk_label, total)
    return [severity_scoring.score_site(h) for h in hits]


def scan_chunks_dask(sequences, labels, guide, max_mismatches=4, client=None,
                     both_strands=True, progress_callback=None):
    guide_codes = gpu.encode_sequence(guide.upper())

    owns_client = client is None
    cluster = None
    if owns_client:
        client, cluster = gpu.create_dask_cluster()

    try:
        from dask.distributed import as_completed as dask_as_completed
        futures = [
            client.submit(scan_chunk, seq, guide_codes, max_mismatches,
                          label, both_strands, pure=False)
            for seq, label in zip(sequences, labels)
        ]
        index_of = {f.key: i for i, f in enumerate(futures)}

        per_chunk = [None] * len(futures)
        done = 0
        for fut in dask_as_completed(futures):
            i = index_of[fut.key]
            per_chunk[i] = fut.result()
            done += 1
            if progress_callback is not None:
                progress_callback(done, len(futures), labels[i], len(per_chunk[i]))

        flat = [hit for chunk_hits in per_chunk for hit in chunk_hits]
        return severity_scoring.rank_off_targets(flat)
    finally:
        if owns_client:
            client.close()
            if cluster is not None:
                cluster.close()


def render_site(hit, width=None):
    guide, site, pam = hit["guide"], hit["site"], hit["pam"]
    bad = {mm["index"] for mm in hit["mismatches"]}

    site_marked = "".join(
        f"[bold red]{base}[/bold red]" if i in bad else base
        for i, base in enumerate(site)
    )
    bar = "".join("[bold red]x[/bold red]" if i in bad else "|" for i in range(len(guide)))
    pam_colour = "green" if severity_scoring.pam_class(pam) == "NGG" else "yellow"

    return [
        f"  guide 5'-{guide}-3'",
        f"  site  5'-{site_marked}-3' [{pam_colour}]{pam}[/{pam_colour}]",
        f"           {bar}",
    ]


SEVERITY_COLOUR = {"HIGH": "bold red", "MEDIUM": "yellow", "LOW": "green"}


def render_ranking(hits, top_n=5):
    if not hits:
        return "No off-target sites found with an NGG/NAG PAM at this mismatch budget."

    lines = []
    for rank, hit in enumerate(hits[:top_n], start=1):
        colour = SEVERITY_COLOUR.get(hit["severity"], "white")
        lines.append(
            f"[{colour}]#{rank}  {hit['severity']:<6}[/{colour}] "
            f"score {hit['severity_score']:.3f}  "
            f"{hit['chunk']}  {hit['strand']}  pos {hit['start']:,}  "
            f"{hit['mismatch_count']} mm  PAM {hit['pam']} ({hit['pam_class']})"
        )
        lines.extend(render_site(hit))
        lines.append("")
    if len(hits) > top_n:
        lines.append(f"  ... {len(hits) - top_n:,} more site(s) below the top {top_n}")
    return "\n".join(lines)
