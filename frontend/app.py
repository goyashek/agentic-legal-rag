"""Streamlit frontend, a thin client over the FastAPI /query endpoint."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")


def query_api(query: str) -> dict[str, Any]:
    """POST one query to the API and return the LegalAdvice payload."""
    request = Request(
        f"{API_URL}/query",
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=90) as response:  # noqa: S310 - API_URL is local config
            return json.loads(response.read())
    except HTTPError as exc:
        detail = exc.read().decode() or exc.reason
        raise RuntimeError(f"The API rejected the query: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Couldn't reach the API at {API_URL}: {exc.reason}") from exc


def render_advice(advice: dict[str, Any]) -> None:
    """Render one LegalAdvice payload."""
    if not advice.get("in_corpus", True):
        st.info("I couldn't match this to the statutes in the current corpus.")
    elif advice.get("confidence") == "low":
        st.warning("This answer is low confidence. Try adding details or naming the offence.")

    st.write(advice["answer"])
    offences = advice.get("offences_identified", [])
    if offences:
        st.caption("Offences identified: " + " · ".join(offences))

    citations = advice.get("citations", [])
    if citations:
        st.subheader("Sources")
        for citation in citations:
            label = f"{citation['act']} section {citation['section_id']}"
            if citation.get("heading"):
                label += f": {citation['heading']}"
            with st.expander(label):
                if citation.get("quote"):
                    st.write(citation["quote"])
                else:
                    st.caption("The answer cites this section from the retrieved corpus.")

    st.caption(advice.get("disclaimer", "This is statutory information, not legal advice."))
    if trace_url := advice.get("trace_url"):
        st.sidebar.link_button("Open LangSmith trace", trace_url)


def main() -> None:
    st.set_page_config(page_title="Agentic Legal RAG: Indian Criminal Law", page_icon="⚖️")
    st.title("⚖️ Agentic Legal RAG")
    st.caption("Indian criminal law (BNS / BNSS / BSA). Statutory information, not legal advice.")

    with st.form("legal-query"):
        query = st.text_area(
            "What happened?",
            placeholder=(
                "Example: Someone took my bicycle without permission. What provision applies?"
            ),
            max_chars=2000,
        )
        submitted = st.form_submit_button("Find relevant sections")

    if submitted:
        if not query.strip():
            st.error("Enter a question first.")
            return
        try:
            with st.spinner("Checking the statutes..."):
                advice = query_api(query.strip())
        except RuntimeError as exc:
            st.error(str(exc))
            return
        render_advice(advice)


if __name__ == "__main__":
    main()
