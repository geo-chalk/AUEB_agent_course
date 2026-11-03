"""Summarise: one product in, one summary out. The page is a client of POST /summarise."""

import httpx
import streamlit as st

from aueb_agents_ca.ui.client import api_health, api_url, make_client, stream_summarise

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

# # What the page is about to do, as the equivalent terminal command: copy it to call the API yourself.
# st.code(
#     f"""curl -s -X POST {api_url()}/summarise -H 'Content-Type: application/json' \\
#   -d '{{"product": "{product}"}}' | jq -r .summary""",
#     language="bash",
# )

if st.button("Summarise", type="primary", disabled=not product):
    # While the agent runs, the API streams its log lines; they show here, then fold away under the result.
    lines: list[str] = []
    result, request_id, error = None, "-", None
    with st.status("Running the agent: searching, reading and analysing reviews...", expanded=True) as status:
        log_box = st.empty()
        try:
            for event in stream_summarise(client, product):
                request_id = event["request_id"]
                if "log" in event:
                    lines.append(event["log"])
                    log_box.code("\n".join(lines), language="log")
                elif "result" in event:
                    result = event["result"]
                elif "error" in event:
                    error = event["error"]
        except httpx.HTTPError as e:
            error = str(e)
        if error or result is None:
            status.update(label="The agent failed", state="error", expanded=True)
        else:
            status.update(
                label=f"Done: {result['steps']} model calls, {len(lines)} log lines (expand to read them)",
                state="complete",
                expanded=False,
            )
    if error or result is None:
        st.error(error or "The API closed the stream without a result.")
        st.stop()
    st.markdown(result["summary"])
    st.caption(f"{result['steps']} model calls · {result['model']} · request `{request_id}`")
    with st.expander("The raw API response"):
        st.json(result)
