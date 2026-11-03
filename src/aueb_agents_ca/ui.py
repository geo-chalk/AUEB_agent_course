"""A small web page for the review summariser. It is a client of the API and knows nothing about the agent:
it sends a product name to /summarise and renders what comes back."""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")  # inside compose: http://api:8000


def api_health(client: httpx.Client) -> dict | None:
    """The API's /health, or None when it cannot be reached."""
    try:
        return client.get("/health", timeout=5).json()
    except httpx.HTTPError:
        return None


def summarise(client: httpx.Client, product: str) -> tuple[dict, str]:
    """Call /summarise and return the response body and the request id the API tagged it with."""
    response = client.post("/summarise", json={"product": product}, timeout=300)  # an agent run takes a while
    response.raise_for_status()
    return response.json(), response.headers.get("X-Request-ID", "-")


def main() -> None:
    st.set_page_config(page_title="Review summariser", page_icon="📝")
    st.title("What do the reviews say?")
    client = httpx.Client(base_url=API_URL)

    health = api_health(client)
    if health is None:
        st.error(f"The API at {API_URL} is not reachable. Start it with `docker compose up api`.")
        st.stop()
    st.caption(f"API: {API_URL} · model: {health['model']}")

    product = st.text_input("Product", placeholder="Sony WH-1000XM5")
    if st.button("Summarise", type="primary", disabled=not product):
        with st.spinner("Searching, reading and analysing reviews (about a minute)..."):
            try:
                result, request_id = summarise(client, product)
            except httpx.HTTPError as e:
                st.error(f"The request failed: {e}")
                st.stop()
        st.markdown(result["summary"])
        st.caption(f"{result['steps']} model calls · {result['model']} · request {request_id}")


if (
    __name__ == "__main__" or st.runtime.exists()
):  # `streamlit run` imports the file; plain python runs main too
    main()
