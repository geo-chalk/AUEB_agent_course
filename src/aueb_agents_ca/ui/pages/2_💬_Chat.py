"""Chat: ask the agent in your own words. Each message is one run of the agent through POST /ask."""

import httpx
import streamlit as st

from aueb_agents_ca.ui.api_client import api_health, api_url, ask, make_client

st.set_page_config(page_title="Chat", page_icon="💬", layout="centered")
st.title("💬 Ask about a product's reviews")
st.caption(
    "Each message is an independent run of the agent: it searches, analyses and answers, then forgets. "
    "Name the product in every message. Questions that are not about product reviews are declined."
)

client = make_client()
health, _ = api_health(client)
if health is None:
    st.error(f"The API at `{api_url()}` is not reachable. Start it with `docker compose up api`.")
    st.stop()

# The chat history lives in the session: Streamlit re-runs this script on every interaction.
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("**Try asking**")
    for suggestion in (
        "What do reviewers dislike about the Sony WH-1000XM5?",
        "Is the Kindle Paperwhite worth it for reading outdoors?",
        "Compare what reviews say about the battery of the AirPods Pro 2 and the WH-1000XM5.",
    ):
        st.code(suggestion, language=None)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

for m in st.session_state.messages:  # redraw the conversation so far
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("caption"):
            st.caption(m["caption"])

if task := st.chat_input("Ask about the reviews of a product"):
    st.session_state.messages.append({"role": "user", "content": task})
    with st.chat_message("user"):
        st.markdown(task)
    with st.chat_message("assistant"):
        with st.spinner("Running the agent (about a minute)..."):
            try:
                result, request_id = ask(client, task)
                answer = result["answer"]
                caption = f"{result['steps']} model calls · request `{request_id}`"
            except httpx.HTTPError as e:
                answer, caption = f"The request failed: {e}", ""
        st.markdown(answer)
        if caption:
            st.caption(caption)
    st.session_state.messages.append({"role": "assistant", "content": answer, "caption": caption})
