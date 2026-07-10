"""My own legal-specific section-precision metrics, run alongside RAGAS.

RAGAS scores answer/context quality generically. These check the thing I actually
care about for a statute system: did I retrieve/cite the right sections? Pure
functions over predicted vs ground-truth section IDs, so they're easy to unit-test
and don't need an LLM.
"""

from __future__ import annotations


def section_precision_at_k(
    retrieved_section_ids: list[str],
    relevant_section_ids: list[str],
    k: int = 5,
) -> float:
    """P@k over section IDs: fraction of top-k retrieved that are truly relevant."""
    raise NotImplementedError("week 3 wed: precision@k over section IDs")


def section_recall(
    retrieved_section_ids: list[str],
    relevant_section_ids: list[str],
) -> float:
    """Recall: fraction of relevant sections that got retrieved at all.

    Honestly the metric that matters most for legal stuff. Missing an applicable
    offence section is way worse than pulling in one extra low-precision one, since
    cross-sectional queries really do need every offence that applies.
    """
    raise NotImplementedError("week 3 wed: recall over section IDs")


def citation_accuracy(
    cited_section_ids: list[str],
    relevant_section_ids: list[str],
) -> float:
    """Of the sections the answer actually CITED, fraction that are truly relevant.

    Different from the deterministic validator (that one checks cited-was-retrieved);
    this one checks cited-was-correct against ground truth.
    """
    raise NotImplementedError("week 3 wed: cited vs ground-truth relevance")
