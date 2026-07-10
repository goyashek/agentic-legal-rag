"""My own legal-specific section-precision metrics, run alongside RAGAS.

RAGAS scores answer/context quality generically. These check the thing I actually
care about for a statute system: did I retrieve/cite the right sections? Pure
functions over predicted vs ground-truth section IDs, so they're easy to unit-test
and don't need an LLM.

Section identity: I compare on (act, section_id) rendered as one string, e.g.
"BNS::103". A bare "103" is ambiguous across acts, so callers should qualify. The
functions themselves just treat the ids as opaque hashable labels, so a caller that
only cares about BNS can pass bare numbers too, as long as it's consistent on both
sides.
"""

from __future__ import annotations

from collections.abc import Sequence


def section_precision_at_k(
    retrieved_section_ids: Sequence[str],
    relevant_section_ids: Sequence[str],
    k: int = 5,
) -> float:
    """P@k over section IDs: fraction of top-k retrieved that are truly relevant.

    Deduplicates the top-k (a section retrieved twice shouldn't inflate the
    denominator) and divides by the number of distinct slots actually filled, so a
    query with fewer than k retrieved chunks isn't unfairly penalised. Returns 0.0
    when nothing was retrieved.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    top_k: list[str] = []
    for sid in retrieved_section_ids:
        if sid not in top_k:
            top_k.append(sid)
        if len(top_k) == k:
            break
    if not top_k:
        return 0.0

    relevant = set(relevant_section_ids)
    hits = sum(1 for sid in top_k if sid in relevant)
    return hits / len(top_k)


def section_recall(
    retrieved_section_ids: Sequence[str],
    relevant_section_ids: Sequence[str],
) -> float:
    """Recall: fraction of relevant sections that got retrieved at all.

    Honestly the metric that matters most for legal stuff. Missing an applicable
    offence section is way worse than pulling in one extra low-precision one, since
    cross-sectional queries really do need every offence that applies. Returns 1.0
    when there are no relevant sections to find (nothing to miss).
    """
    relevant = set(relevant_section_ids)
    if not relevant:
        return 1.0
    retrieved = set(retrieved_section_ids)
    return len(relevant & retrieved) / len(relevant)


def citation_accuracy(
    cited_section_ids: Sequence[str],
    relevant_section_ids: Sequence[str],
) -> float:
    """Of the sections the answer actually CITED, fraction that are truly relevant.

    Different from the deterministic validator (that one checks cited-was-retrieved);
    this one checks cited-was-correct against ground truth. Returns 1.0 when nothing
    was cited (vacuously precise — no wrong citations were made).
    """
    cited = set(cited_section_ids)
    if not cited:
        return 1.0
    relevant = set(relevant_section_ids)
    return len(cited & relevant) / len(cited)
