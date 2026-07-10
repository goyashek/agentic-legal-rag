"""Tests for the metadata enrichment layer (stage 3).

Tiers, same as the other ingest tests:
  - Pure-function tests for _classify_cell, _aggregate, _band and the enrich() join
    logic over synthetic chunks. No PDFs, always run, CI-safe. These pin the
    legal-data discipline: conditional/conflicting -> None, never a guessed flag.
  - Integration tests parse the real PDFs and assert known offences classify
    correctly (BNS 103 murder = cognizable + non-bailable, 103 -> IPC 302). Skip
    cleanly when the source PDFs aren't present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingest.chunk_chonkie import LegalChunk
from src.ingest.enrich_metadata import (
    _aggregate,
    _band,
    _classify_cell,
    enrich,
    load_chapter_titles,
    load_ipc_bns_mapping,
    load_offence_classification,
)

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
BNSS = RAW / "bnss.pdf"
BNS = RAW / "bns.pdf"
COMPARISON = RAW / "COMPARISON SUMMARY BNS to IPC .pdf"


class TestClassifyCell:
    def test_plain_cognizable_true(self) -> None:
        assert _classify_cell("Cognizable.", "cogn") is True

    def test_non_cognizable_false(self) -> None:
        assert _classify_cell("Non-cognizable.", "cogn") is False

    def test_drifted_non_prefix_still_false(self) -> None:
        """Order-independent: "cognizable Non-" (prefix sorted after) must read False."""
        assert _classify_cell("cognizable non-", "cogn") is False

    def test_conditional_is_none(self) -> None:
        assert _classify_cell("According as offence abetted is cognizable", "cogn") is None

    def test_blank_is_none(self) -> None:
        assert _classify_cell("   ", "bail") is None

    def test_wrong_stem_is_none(self) -> None:
        """A bailable cell that only says 'triable' yields None, not a false read."""
        assert _classify_cell("Court of Session.", "bail") is None

    def test_bailable_true_and_false(self) -> None:
        assert _classify_cell("Bailable.", "bail") is True
        assert _classify_cell("Non-bailable.", "bail") is False


class TestAggregate:
    def test_unanimous_true(self) -> None:
        assert _aggregate([True, True, None]) is True

    def test_conflict_is_none(self) -> None:
        assert _aggregate([True, False]) is None

    def test_all_none_is_none(self) -> None:
        assert _aggregate([None, None]) is None

    def test_single_value(self) -> None:
        assert _aggregate([False]) is False


class TestBand:
    def test_bands_are_ordered(self) -> None:
        assert _band(72) == "sec"
        assert _band(150) == "mid"
        assert _band(305) == "cogn"
        assert _band(372) == "bail"
        assert _band(459) == "court"


class TestEnrichJoin:
    """enrich() join logic over synthetic chunks + pre-loaded mapping (no PDFs)."""

    def _chunks(self) -> list[LegalChunk]:
        return [
            LegalChunk("BNS::103::0", "BNS", "103", "Punishment for murder", "t", chapter="VI"),
            LegalChunk("BNSS::35::0", "BNSS", "35", "Arrest", "t", chapter="V"),
        ]

    def test_ipc_equivalents_only_on_bns(self) -> None:
        chunks = self._chunks()
        enrich(chunks, ipc_bns_mapping={"302": "103", "379": "303"})
        bns = chunks[0].metadata["ipc_equivalents"]
        bnss = chunks[1].metadata["ipc_equivalents"]
        assert bns == ["302"]
        assert bnss == []

    def test_flags_default_none_without_source(self) -> None:
        chunks = self._chunks()
        enrich(chunks)  # no bnss_pdf
        assert chunks[0].metadata["cognizable"] is None
        assert chunks[0].metadata["bailable"] is None

    def test_category_none_without_titles(self) -> None:
        chunks = self._chunks()
        enrich(chunks)
        assert chunks[0].metadata["offence_category"] is None

    def test_multiple_ipc_sorted(self) -> None:
        chunks = [LegalChunk("BNS::318::0", "BNS", "318", "Cheating", "t", chapter="X")]
        enrich(chunks, ipc_bns_mapping={"420": "318", "415": "318", "417": "318"})
        assert chunks[0].metadata["ipc_equivalents"] == ["415", "417", "420"]


@pytest.mark.skipif(not BNSS.exists(), reason="source PDFs not present")
class TestRealClassification:
    def test_landmark_offences_classify_correctly(self) -> None:
        cls = load_offence_classification(BNSS)
        # murder: cognizable, non-bailable
        assert cls["103"] == {"cognizable": True, "bailable": False}
        # theft (303): cognizable, non-bailable
        assert cls["303"]["cognizable"] is True

    def test_conditional_rows_stay_none(self) -> None:
        """Abetment sections carry "According as offence abetted is..." -> None."""
        cls = load_offence_classification(BNSS)
        # section 49 is abetment; its bailable is conditional
        if "49" in cls:
            assert cls["49"]["bailable"] is None

    def test_no_false_cognizable(self) -> None:
        """Every classified value is a real bool or None (never a stray string)."""
        cls = load_offence_classification(BNSS)
        for flags in cls.values():
            assert flags["cognizable"] in (True, False, None)
            assert flags["bailable"] in (True, False, None)


@pytest.mark.skipif(not COMPARISON.exists(), reason="comparison PDF not present")
class TestRealIpcMapping:
    def test_landmark_ipc_to_bns(self) -> None:
        m = load_ipc_bns_mapping(COMPARISON)
        assert m["302"] == "103"   # murder
        assert m["379"] == "303"   # theft
        assert m["420"] == "318"   # cheating
        assert m["375"] == "63"    # rape (definition)
        assert m["376"] == "64"    # rape (punishment)

    def test_mapping_is_substantial(self) -> None:
        m = load_ipc_bns_mapping(COMPARISON)
        assert len(m) > 400


@pytest.mark.skipif(not BNS.exists(), reason="source PDFs not present")
class TestRealChapterTitles:
    def test_known_chapter_titles(self) -> None:
        titles = load_chapter_titles(BNS)
        assert titles["VI"] == "Of Offences Affecting The Human Body"
        assert "I" in titles
        assert len(titles) >= 18  # BNS has 20 chapters
