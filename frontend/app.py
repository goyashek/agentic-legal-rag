"""Streamlit frontend, a thin client over the FastAPI /query endpoint.

No business logic here, it just POSTs to the API and renders the LegalAdvice.
On the page: a query input box, expandable cards per cited section, offence chips,
low-confidence / not-in-corpus banners, and a LangSmith trace link in the sidebar
so a run is auditable.
"""

from __future__ import annotations

import os

import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")


def query_api(query: str) -> dict:
    """POST the query to the API, return the LegalAdvice payload as a dict."""
    raise NotImplementedError("week 4 tue: requests.post(f'{API_URL}/query', ...)")


def render_advice(advice: dict) -> None:
    """Render a LegalAdvice payload: answer, citation cards, chips, banners, trace link."""
    raise NotImplementedError("week 4 tue: streamlit layout")


def main() -> None:
    st.set_page_config(page_title="Agentic Legal RAG: Indian Criminal Law", page_icon="⚖️")
    st.title("⚖️ Agentic Legal RAG")
    st.caption("Indian criminal law (BNS / BNSS / BSA). Statutory information, not legal advice.")
    raise NotImplementedError("week 4 tue: input box, call query_api, render_advice")


if __name__ == "__main__":
    main()
