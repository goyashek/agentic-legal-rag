"""Tests for the ingestion layer (stage 1: PDF -> RawSection).

Two tiers:
  - Pure-function tests for verify_section_counts (the delta gate). No I/O, always run,
    CI-safe.
  - Integration tests that parse the real BNS/BNSS/BSA PDFs in data/raw/. These skip
    cleanly when the PDFs aren't present (they're git-ignored, licensing), so CI stays
    green without them, but they're the real proof the parser hits the published counts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingest.parse_pdf import (
    PUBLISHED_SECTION_COUNTS,
    RawSection,
    parse_statute,
    verify_section_counts,
)

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
PDFS = {"BNS": RAW / "bns.pdf", "BNSS": RAW / "bnss.pdf", "BSA": RAW / "bsa.pdf"}


class TestVerifySectionCounts:
    """The count gate is pure and deterministic, so it gets the hardest tests."""

    def test_clean_parse_is_all_zero(self) -> None:
        secs = [RawSection("BNS", str(i), "h", "body") for i in range(1, 359)]
        assert verify_section_counts(secs, {"BNS": 358}) == {"BNS": 0}

    def test_reports_signed_delta(self) -> None:
        secs = [RawSection("BSA", str(i), "h", "body") for i in range(1, 169)]  # 168
        assert verify_section_counts(secs, {"BSA": 170}) == {"BSA": -2}

    def test_over_count_is_positive(self) -> None:
        secs = [RawSection("BSA", str(i), "h", "body") for i in range(1, 173)]  # 172
        assert verify_section_counts(secs, {"BSA": 170}) == {"BSA": 2}

    def test_missing_act_counts_as_zero_parsed(self) -> None:
        assert verify_section_counts([], {"BNS": 358}) == {"BNS": -358}


def _require(act: str) -> Path:
    path = PDFS[act]
    if not path.exists():
        pytest.skip(f"{path} not present (git-ignored source PDF); drop it in to run")
    return path


@pytest.mark.parametrize("act", ["BNS", "BNSS", "BSA"])
class TestParseRealStatutes:
    def test_section_count_matches_published_total(self, act: str) -> None:
        secs = parse_statute(_require(act), act)
        assert len(secs) == PUBLISHED_SECTION_COUNTS[act]

    def test_section_ids_unique(self, act: str) -> None:
        secs = parse_statute(_require(act), act)
        ids = [s.section_id for s in secs]
        assert len(ids) == len(set(ids))

    def test_numbering_is_contiguous_1_to_n(self, act: str) -> None:
        """Every integer 1..N is present (letter-suffixed ids collapse to their number)."""
        import re

        secs = parse_statute(_require(act), act)
        nums = {int(re.sub(r"[A-Z]", "", s.section_id)) for s in secs}
        expected = set(range(1, PUBLISHED_SECTION_COUNTS[act] + 1))
        assert expected - nums == set()

    def test_every_section_has_heading_and_body(self, act: str) -> None:
        secs = parse_statute(_require(act), act)
        assert all(len(s.heading.strip()) >= 3 for s in secs)
        assert all(len(s.text.strip()) >= 15 for s in secs)

    def test_every_section_tagged_with_chapter(self, act: str) -> None:
        secs = parse_statute(_require(act), act)
        assert all(s.chapter for s in secs)

    def test_no_section_body_leaks_into_next(self, act: str) -> None:
        """A body shouldn't contain the next section's numbered dash-heading start."""
        import re

        secs = parse_statute(_require(act), act)
        leak = re.compile(r"(?m)^\s*\d+[A-Z]?\.\s+\w.{0,60}(?:—|–|--)")
        offenders = [s.section_id for s in secs if leak.search(s.text)]
        assert offenders == []


class TestKnownContent:
    """Spot-check that famous sections parse with the right heading + body."""

    @pytest.mark.parametrize(
        ("act", "section_id", "keyword"),
        [
            ("BNS", "103", "murder"),        # punishment for murder
            ("BNS", "63", "rape"),           # rape defined
            ("BNSS", "173", "information"),  # information in cognizable cases
            ("BSA", "3", "evidence"),        # evidence may be given of facts in issue
        ],
    )
    def test_landmark_section(self, act: str, section_id: str, keyword: str) -> None:
        secs = parse_statute(_require(act), act)
        match = next((s for s in secs if s.section_id == section_id), None)
        assert match is not None, f"{act} s.{section_id} not parsed"
        assert keyword.lower() in (match.heading + " " + match.text).lower()

    def test_last_section_stops_before_schedule(self) -> None:
        """BNSS s.531 must not swallow the trailing schedule of forms (was 197KB)."""
        secs = parse_statute(_require("BNSS"), "BNSS")
        last = next(s for s in secs if s.section_id == "531")
        assert "Seal of the Court" not in last.text
        assert len(last.text) < 20_000
