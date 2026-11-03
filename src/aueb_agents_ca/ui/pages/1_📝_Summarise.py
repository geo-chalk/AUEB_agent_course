"""Summarise: one product in, one summary out. The page is a client of POST /summarise."""

import httpx
import streamlit as st

from aueb_agents_ca.ui.client import api_health, api_url, make_client, summarise

st.set_page_config(page_title="Summarise", page_icon="📝", layout="centered")
st.title("📝 Summarise the reviews of a product")

# Products with plenty of reviews online: good first runs. Type anything else in the box.
EXAMPLES = [
    "Sony WH-1000XM5",
    "Apple AirPods Pro 2",
    "Kindle Paperwhite",
    "Dyson V15 Detect",
    "Nintendo Switch OLED",
    "Logitech MX Master 3S",
]

client = make_client()
health, _ = api_health(client)
if health is None:
    st.error(f"The API at `{api_url()}` is not reachable. Start it with `docker compose up api`.")
    st.stop()

example = st.selectbox("Try one of these", EXAMPLES)
product = st.text_input("Product", value=example)

# What the page is about to do, as the equivalent terminal command: copy it to call the API yourself.
st.code(
    f"""curl -s -X POST {api_url()}/summarise -H 'Content-Type: application/json' \\
  -d '{{"product": "{product}"}}' | jq -r .summary""",
    language="bash",
)

if st.button("Summarise", type="primary", disabled=not product):
    with st.spinner("Searching, reading and analysing reviews (about a minute, ~20 model calls)..."):
        try:
            result, request_id = summarise(client, product)
        except httpx.HTTPError as e:
            st.error(f"The request failed: {e}")
            st.stop()
    st.markdown(result["summary"])
    st.caption(f"{result['steps']} model calls · {result['model']} · request `{request_id}`")
    with st.expander("The raw API response"):
        st.json(result)
