"""Summarise: one product in, one summary out. The page is a client of POST /summarise."""

import re

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
health, _ = api_health(client=client)
if health is None:
    st.error(f"The API at `{api_url()}` is not reachable. Start it with `docker compose up api`.")
    st.stop()

example = st.selectbox("Try one of these", EXAMPLES)
product = st.text_input("Product", value=example)
# The loop's safety net: the agent usually finishes in 3 or 4 model calls; fewer steps cut a run short.
max_steps = st.slider("Max model calls (max_steps)", min_value=1, max_value=20, value=8)

# # What the page is about to do, as the equivalent terminal command: copy it to call the API yourself.
# st.code(
#     f"""curl -s -X POST {api_url()}/summarise -H 'Content-Type: application/json' \\
#   -d '{{"product": "{product}", "max_steps": {max_steps}}}' | jq -r .summary""",
#     language="bash",
# )

if st.button("Summarise", type="primary", disabled=not product):
    # While the agent runs, the API streams its log lines; they show here, then fold away under the result.
    lines: list[str] = []
    result, request_id, error = None, "-", None
    with st.status("Running the agent: searching, reading and analysing reviews...", expanded=True) as status:
        log_box = st.empty()
        try:
            for event in stream_summarise(client=client, product=product, max_steps=max_steps):
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
    if result.get("image"):
        image = result["image"]
        # A small picture beside the product name. The thumbnail is served by the search engine and
        # always loads; the original ("url") often refuses to be shown inside another site.
        picture, heading = st.columns([1, 5])
        picture.image(image["thumbnail"], width=110)
        heading.subheader(result["product"])
        heading.caption(f"[{image['title']}]({image['source']})")
    # Safety net: a model sometimes pastes an image into its Markdown; the page shows the picture itself.
    st.markdown(re.sub(r"!\[[^\]]*\]\([^)]*\)", "", result["summary"]))
    st.caption(f"{result['steps']} model calls · {result['model']} · request `{request_id}`")
    with st.expander("The raw API response"):
        st.json(result)
