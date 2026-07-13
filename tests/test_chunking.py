"""Tests for the chunking layer (stage 2: RawSection -> LegalChunk).

Tiers, same idea as test_ingest:
  - Pure-function tests (write_chunks_jsonl, _heading_summary, chunk_id shape) run
    always, no model, no PDFs, CI-safe.
  - chunk_sections tests need Chonkie + the potion-base-8M model. They run on small
    SYNTHETIC RawSections (no PDF needed) so they exercise the keep-whole-vs-split
    logic fast, and skip cleanly if Chonkie isn't importable.
  - One integration test parses the real PDFs end-to-end; skips when PDFs absent.

The behaviour I most want pinned: a section that fits the token budget stays ONE
chunk (regression guard — the semantic chunker was shattering BNS s.1 into 5 scraps),
and every chunk_id is unique and traces back to exactly one (act, section_id).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ingest.chunk_chonkie import (
    LegalChunk,
    _heading_summary,
    chunk_sections,
    write_chunks_jsonl,
)
from src.ingest.parse_pdf import RawSection, parse_statute

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

chonkie = pytest.importorskip("chonkie", reason="chonkie not installed; run `uv sync`")

# One short section (fits budget -> 1 chunk) and one long one (overflows -> splits).
SHORT = RawSection("BNS", "103", "Punishment for murder",
                   "Whoever commits murder shall be punished with death or imprisonment "
                   "for life, and shall also be liable to fine.", chapter="VI")
LONG = RawSection("BNS", "999", "Synthetic long section",
                  ("Distinct legal concept about property. " * 40)
                  + ("Unrelated concept about marriage and family relations. " * 40)
                  + ("A third theme entirely, concerning evidence and documents. " * 40),
                  chapter="I")


class TestHeadingSummary:
    def test_returns_trimmed_heading(self) -> None:
        assert _heading_summary("body", "  Punishment for murder.  ") == "Punishment for murder"

    def test_ignores_body(self) -> None:
        assert _heading_summary("anything at all", "Theft") == "Theft"


class TestWriteChunksJsonl:
    def test_writes_one_json_object_per_line(self, tmp_path: Path) -> None:
        chunks = [
            LegalChunk("BNS::1::0", "BNS", "1", "Short title", "text a", summary="s"),
            LegalChunk("BNS::2::0", "BNS", "2", "Definitions", "text b"),
        ]
        out = tmp_path / "sections.jsonl"
        n = write_chunks_jsonl(chunks, out)
        assert n == 2
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["chunk_id"] == "BNS::1::0"
        assert first["act"] == "BNS"
        assert first["summary"] == "s"

    def test_creates_parent_dir(self, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "deep" / "sections.jsonl"
        write_chunks_jsonl([LegalChunk("BNS::1::0", "BNS", "1", "h", "t")], out)
        assert out.exists()

    def test_roundtrips_unicode(self, tmp_path: Path) -> None:
        chunks = [LegalChunk("BNS::1::0", "BNS", "1", "h", "Sanhita § 103 — dash")]
        out = tmp_path / "s.jsonl"
        write_chunks_jsonl(chunks, out)
        assert json.loads(out.read_text(encoding="utf-8"))["text"] == "Sanhita § 103 — dash"


class TestChunkSectionsSynthetic:
    def test_short_section_stays_one_chunk(self) -> None:
        chunks = chunk_sections([SHORT])
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "BNS::103::0"

    def test_long_section_splits(self) -> None:
        chunks = chunk_sections([LONG], max_tokens=128)
        assert len(chunks) > 1
        assert all(c.section_id == "999" for c in chunks)
        # ids are contiguous 0..n-1
        assert [c.chunk_id for c in chunks] == [f"BNS::999::{i}" for i in range(len(chunks))]

    def test_summary_prepended_by_default(self) -> None:
        chunk = chunk_sections([SHORT])[0]
        assert chunk.summary == "Punishment for murder"
        assert chunk.text.startswith("Punishment for murder\n\n")

    def test_summary_augment_off(self) -> None:
        chunk = chunk_sections([SHORT], summary_augment=False)[0]
        assert chunk.summary == ""
        assert chunk.text.startswith("Whoever commits murder")

    def test_custom_summarizer_injected(self) -> None:
        chunk = chunk_sections([SHORT], summarizer=lambda t, h: "CUSTOM")[0]
        assert chunk.summary == "CUSTOM"
        assert chunk.text.startswith("CUSTOM\n\n")

    def test_chapter_carried_through(self) -> None:
        assert chunk_sections([SHORT])[0].chapter == "VI"


@pytest.mark.skipif(not (RAW / "bns.pdf").exists(), reason="source PDFs not present")
class TestChunkRealCorpus:
    def test_pipeline_produces_unique_ids_and_no_lost_sections(self) -> None:
        secs: list[RawSection] = []
        for act in ("BNS", "BNSS", "BSA"):
            secs += parse_statute(RAW / f"{act.lower()}.pdf", act)
        chunks = chunk_sections(secs)

        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "duplicate chunk_id"

        # every parsed section appears in >=1 chunk
        sec_keys = {(s.act, s.section_id) for s in secs}
        chunk_keys = {(c.act, c.section_id) for c in chunks}
        assert sec_keys - chunk_keys == set(), "some sections produced no chunk"

    def test_most_sections_stay_whole(self) -> None:
        """Regression guard: sub-512-tok sections shouldn't over-split. Expect the
        large majority of sections to be a single chunk."""
        secs: list[RawSection] = []
        for act in ("BNS", "BNSS", "BSA"):
            secs += parse_statute(RAW / f"{act.lower()}.pdf", act)
        chunks = chunk_sections(secs)
        from collections import Counter

        per_sec = Counter((c.act, c.section_id) for c in chunks)
        single = sum(1 for v in per_sec.values() if v == 1)
        assert single / len(per_sec) > 0.85, f"only {single}/{len(per_sec)} sections stayed whole"

    def test_bns_303_base_punishment_stays_in_one_chunk(self) -> None:
        section = next(s for s in parse_statute(RAW / "bns.pdf", "BNS") if s.section_id == "303")
        chunks = chunk_sections([section])
        assert any(
            "shall be punished with imprisonment of either description for a term"
            " which may extend to three years" in " ".join(c.text.split())
            for c in chunks
        )
