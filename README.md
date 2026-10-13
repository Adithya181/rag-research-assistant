# 🤖 Multi-Tool RAG Research Assistant

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://rag-research-assistant-adithya.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-ff4b4b)](https://streamlit.io/)

🔗 **[Live Demo](https://rag-research-assistant-adithya.streamlit.app)**

A Retrieval-Augmented Generation (RAG) system over 8 foundational machine learning papers (Transformer, BERT, ResNet, GPT-3, Adam, Dropout, Batch Normalization, ViT), wrapped in a multi-tool LangChain agent and served through an interactive Streamlit chat UI.

Ask it to explain a paper's architecture, run a calculation, extract keywords, or count words in a passage — the agent uses LLM-driven tool selection to pick the right tool and answers accordingly, citing sources for anything grounded in the papers.

---

## Table of contents

- [Features](#features)
- [Retrieval evaluation](#retrieval-evaluation)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Deploying to Streamlit Community Cloud](#deploying-to-streamlit-community-cloud)
- [Example queries](#example-queries)
- [Tech stack](#tech-stack)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features

- **RAG pipeline**: PDF ingestion → overlapping chunking → `all-MiniLM-L6-v2` sentence embeddings → FAISS similarity search → grounded answer generation via Groq (`openai/gpt-oss-120b`).
- **LLM-driven tool-calling agent**: routes each query to whichever tool fits using LangChain's `create_tool_calling_agent` — no regex, no keyword matching, the model itself decides.
- **Relevance-filtered retrieval**: chunks below a similarity threshold (tuned to 0.10) are dropped before being sent to the LLM, reducing hallucinated answers on out-of-scope questions.
- **Single-pass retrieval**: retrieval runs exactly once per question — the same document set is used both for context and for building the sources list (no double embedding cost).
- **Auto-rebuilding index**: a checksum of the PDFs folder is stored next to the index; if you add or replace papers, the index rebuilds automatically on next startup.
- **Chat memory**: prior turns are passed to the agent as `chat_history`, so follow-up questions like "explain that more simply" work correctly.
- **Safe calculator**: arithmetic uses an AST-based evaluator — no `eval()`, no code injection, exponents capped to prevent denial-of-service.
- **Streamlit chat UI**: persistent chat history, source citations, tool-type badges, and a per-session request limit to protect the shared Groq API key.

## Retrieval evaluation

Measured on a 50-question golden set covering all 8 papers (`tests/golden_set.json`).

| Config | hit-rate@1 | hit-rate@4 | MRR |
|---|---|---|---|
| threshold=0.30 (original) | 80% | 82% | 0.807 |
| **threshold=0.10 (tuned)** | **86%** | **98%** | **0.912** |

Threshold was chosen by grid search (`python scripts/evaluate_retrieval.py --tune`).
Lowering from 0.30 → 0.10 improved hit-rate@4 from **82% → 98%** (+16 pp) and MRR from **0.807 → 0.912** (+0.105).
The single remaining miss (Q38, CLS token in ViT) is expected: both BERT and ViT use CLS tokens, so BERT chunks rank slightly higher.

## Architecture

```
User query
    │
    ▼
┌───────────────────────────────────────────────────┐
│  LangChain Tool-Calling Agent (gpt-oss-20b)        │
│  Receives chat history → selects tool via LLM      │
└───────┬──────────────┬────────────┬────────────────┘
        │              │            │            │
        ▼              ▼            ▼            ▼
  RAGEngine       Calculator   Keyword      Word
  (gpt-oss-120b)  (AST eval)   Extractor    Counter
   1. embed                    (TF-ranked)  (words +
   2. FAISS search                          sentences +
   3. filter ≥0.3                           chars)
   4. generate
```

## Project structure

```
rag-research-assistant/
├── app.py                          # Streamlit chat UI
├── src/
│   ├── rag_engine.py               # ingestion, chunking, embeddings, FAISS, generation
│   └── agent.py                    # LangChain tool-calling agent + safe calculator
├── scripts/
│   ├── download_papers.py          # reproducibly fetches the 8 papers from arXiv
│   ├── debug_retrieval.py          # diagnostic: inspect index sources and scores
│   └── evaluate_retrieval.py       # retrieval evaluation & threshold tuning
├── tests/
│   ├── golden_set.json             # 50-question retrieval benchmark
│   ├── test_calculator.py          # unit tests for safe arithmetic evaluator
│   └── test_tools.py               # unit tests for keyword extractor & word counter
├── results/
│   ├── eval_baseline.json          # baseline retrieval evaluation metrics
│   └── eval_after_tuning.json      # post-tuning retrieval evaluation metrics
├── data/
│   ├── papers/                     # PDFs (committed)
│   └── faiss_index/                # persisted FAISS index (committed)
├── requirements.txt
├── runtime.txt
└── LICENSE
```

## Setup

1. **Clone and create a virtual environment**

   ```bash
   git clone https://github.com/Adithya181/rag-research-assistant.git
   cd rag-research-assistant
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Get a free Groq API key** at [console.groq.com/keys](https://console.groq.com/keys)

3. **Add your key locally** (never commit this file):

   ```bash
   mkdir -p .streamlit
   echo 'GROQ_API_KEY = "gsk_..."' > .streamlit/secrets.toml
   ```

4. **(Optional) Re-download papers** — PDFs are already committed; only needed if you want a fresh copy:

   ```bash
   python scripts/download_papers.py
   ```

5. **Run the app**

   ```bash
   streamlit run app.py
   ```

6. **Run tests**

   ```bash
   python -m pytest tests/ -v
   ```

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing at your repo and `app.py`.
3. In the app's **Settings → Secrets**, add:

   ```toml
   GROQ_API_KEY = "your-real-key"
   ```

The PDFs and FAISS index are committed, so no build step is needed on Streamlit Cloud.

## Example queries

| Query | Tool used |
|---|---|
| "What is the attention mechanism in transformers?" | RAG (knowledge_base_search) |
| "Explain how ResNet solves the vanishing gradient problem" | RAG (knowledge_base_search) |
| "calculate 50+6\*7%10-5" | Calculator |
| "what is 2**10" | Calculator |
| "extract keywords from Artificial Intelligence is transforming industries" | Keyword extractor |
| "word count of The quick brown fox jumps over the lazy dog" | Word counter |
| "what is overfitting?" | Direct LLM answer (general ML knowledge) |

## Tech stack

| Component | Library / Service |
|---|---|
| LLM (generation) | Groq `openai/gpt-oss-120b` |
| LLM (agent routing) | Groq `openai/gpt-oss-20b` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS (`faiss-cpu`) |
| Orchestration | LangChain 0.3 (`create_tool_calling_agent`) |
| PDF loading | `pypdf` via `PyPDFDirectoryLoader` |
| UI | Streamlit 1.45+ |

## Troubleshooting

- **`GROQ_API_KEY` not found** — make sure `.streamlit/secrets.toml` exists locally (or the key is set under Settings → Secrets on Streamlit Cloud) and the key name matches exactly.
- **Index looks stale after changing papers** — the app detects PDF changes via checksum and rebuilds automatically. You can also force a rebuild by deleting `data/faiss_index/`.
- **Slow first response** — the first query loads the embedding model into memory; subsequent queries are much faster.
- **Rate limit errors** — you've hit your Groq per-minute or daily quota. Wait a minute and retry, or check [console.groq.com/settings/limits](https://console.groq.com/settings/limits).

## Contributing

Issues and pull requests are welcome. If you're adding a new tool to the agent, please:

1. Define it with `@tool` inside `build_agent()` in `src/agent.py`.
2. Add it to the `tools` list and describe its routing trigger in the system prompt.
3. Add unit tests in `tests/`.

## License

[MIT](LICENSE)
