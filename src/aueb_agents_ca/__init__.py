"""A small agent that summarises the reviews of a product. Read the modules in this order:

    settings.py  configuration, read once from the environment (.env) and passed to whoever needs it
    log.py       logging setup: one line per event, tagged with a request id
    tools.py     the agent's tools (search, analyse) and the JSON schemas the model sees
    agent.py     the loop: send the conversation, run the tools the model asks for, repeat
    api.py       the agent as a web service (FastAPI)

Each module depends only on the ones above it. The notebooks under notebooks/ tell the same story in the
same order, one lesson per file.
"""
