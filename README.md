# Semantic QA over chat conversation

This repo lets you ask a sample chat conversation questions using Langchain

## How to use

1. Add a .env file:

```
OPENAI_API_KEY=
ACTIVELOOP_TOKEN=
ACTIVELOOP_ORG=
```

2. Run `python3 chat/ingest.py` to upload all the messages in messages.txt to DeepLake store
3. Run `python3 chat/ask.py` to query the messages
