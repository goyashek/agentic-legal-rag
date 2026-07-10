"""Stage 3 of ingestion: attach only the metadata the agent actually routes on.

I kept scope tight here. The agent filters on offence category and the
cognizable/bailable flags, and the IPC->BNS bridge feeds both the fast path and
the AIBE eval, so those fields earn their place. Anything else is just noise.
"""

from __future__ import annotations

from pathlib import Path

from src.ingest.chunk_chonkie import LegalChunk


def load_ipc_bns_mapping(mapping_csv: str | Path) -> dict[str, str]:
    """Load the IPC->BNS section mapping, e.g. {"302": "103", "379": "303(2)"}.

    The fast path uses it to resolve "302 IPC" -> BNS 103, and aibe_eval uses it
    because a lot of AIBE questions still cite repealed IPC, so I map them onto
    the BNS corpus.
    """
    raise NotImplementedError("week 1 wed: parse the mapping csv")


def enrich(
    chunks: list[LegalChunk],
    *,
    metadata_csv: str | Path | None = None,
    ipc_bns_mapping: dict[str, str] | None = None,
) -> list[LegalChunk]:
    """Fill chunk.metadata with the routing/filtering fields.

    Writes offence_category (str), cognizable/bailable (bool | None), and
    ipc_equivalents (the IPC sections that map to this BNS section). metadata_csv
    is my section -> {category, cognizable, bailable} table, and ipc_bns_mapping
    is reverse-applied to tag each BNS chunk. Missing values stay None, not guessed.
    """
    raise NotImplementedError("week 1 wed: join metadata csv + reverse ipc mapping")
