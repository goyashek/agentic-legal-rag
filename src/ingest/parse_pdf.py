"""Stage 1 of ingestion: parse BNS/BNSS/BSA gazette PDFs into raw sections.

Output feeds chunk_chonkie.chunk_sections. I wrote the parsing myself instead of
using a framework because the legal structure matters too much to hand off. I do
the Act -> Chapter -> Section hierarchy parse here to fix metadata boundaries
before Chonkie runs semantic chunking, so a bad split can't silently glue two
offences together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RawSection:
    """One statutory section lifted from the PDF, before chunking.

    section_id has to be unique within an act, and text needs to hold the full
    section body (sub-clauses included) without truncating mid-clause or leaking
    into the next section.
    """

    act: str                      # "BNS" | "BNSS" | "BSA"
    section_id: str               # "103", "103(1)", "63A" as printed
    heading: str                  # marginal/section heading
    text: str                     # full section body
    chapter: str | None = None    # chapter number/title if present
    page_start: int | None = None
    page_end: int | None = None
    sub_clauses: list[str] = field(default_factory=list)


def parse_statute(pdf_path: str | Path, act: str) -> list[RawSection]:
    """Extract all sections from one statute PDF, one RawSection each, in order.

    act is the statute code ("BNS", "BNSS", or "BSA"). The parsed count should
    match the published section count and section_ids stay unique, so I log a
    warning on mismatch instead of passing silently.
    """
    raise NotImplementedError("week 1 tue: implement with pymupdf hierarchy parse")


def verify_section_counts(sections: list[RawSection], expected: dict[str, int]) -> dict[str, int]:
    """Check parsed section counts per act against published totals.

    expected is like {"BNS": 358, "BNSS": ..., "BSA": ...}. Returns {act: delta}
    where delta = parsed_count - expected_count, so all-zero means clean.
    """
    raise NotImplementedError("week 1 mon: the section-count sanity gate")
