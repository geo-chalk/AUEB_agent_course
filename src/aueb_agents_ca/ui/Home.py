"""Home: is the API up, and what is it running? A health dashboard, and the way to the other pages."""

import streamlit as st

from aueb_agents_ca.ui.api_client import api_health, api_url, make_client

st.set_page_config(page_title="Review summariser", page_icon="📝", layout="centered")
st.title("📝 Review summariser")
st.markdown("""
Welcome. Use the sidebar to navigate:
- **Summarise**: pick a product (or type one) and get a summary of its reviews.
- **Chat**: ask the agent questions about a product's reviews.

This page checks that the API behind both is reachable and shows what it is running.
""")

st.subheader("📊 API health")
client = make_client()
health, latency_ms = api_health(client)

if health is None:
    st.error(
        f"The API at `{api_url()}` is not reachable. Start it with `docker compose up api` and press *Check again*."
    )
else:
    col1, col2, col3 = st.columns(3)
    col1.metric("Status", health["status"].upper())
    col2.metric("Model", health["model"])
    col3.metric("Latency", f"{latency_ms:.0f} ms")
    st.markdown(f"""
| | |
|---|---|
| API | `{api_url()}` |
| Embedding model | `{health.get("embedding_model") or "none (aspect merging off)"}` |
| Tools the agent can call | `{"`, `".join(health.get("tools", []))}` |
| Interactive docs | [{api_url()}/docs]({api_url()}/docs) |
""")
    st.caption("The health check makes no model call, so it costs nothing to press.")

st.button("Check again")  # any button press re-runs the page, and the page re-checks
