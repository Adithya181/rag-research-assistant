# 🤖 Multi-Tool RAG Research Assistant

[![Live Demo](https://rag-research-assistant-adithya.streamlit.app/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](#license)

A Retrieval-Augmented Generation (RAG) system over 8 foundational machine
learning papers (Transformer, BERT, ResNet, GPT-3, Adam, Dropout, Batch
Normalization, ViT), wrapped in a multi-tool agent and served through an
interactive Streamlit chat UI.

Ask it to explain a paper's architecture, run a calculation, extract
keywords, or count words in a passage — the agent decides which tool to
use and answers accordingly, citing sources for anything grounded in the
papers.

🔗 **[Try the live demo](https://rag-research-assistant-fuzjfaqbygqztvishbrv6c.streamlit.app)**

---

## Table of contents

- [Features](#features)
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

- **RAG pipeline**: PDF ingestion → overlapping chunking → `all-MiniLM-L6-v2` sentence embeddings → FAISS similarity search → grounded answer generation via a Groq-hosted LLaMA 3.1 model.
- **Multi-tool agent**: routes each query to whichever tool fits — RAG, calculator, keyword extractor, or word counter — with schema validation and an execution log.
- **Distance-filtered retrieval**: chunks beyond a similarity threshold are dropped instead of being forced into the prompt, reducing hallucinated "answers" to out-of-scope questions.
- **Streamlit chat UI**: persistent chat history, source citations, tool-type badges, and a sidebar explaining what the agent can do.

## Architecture

```
User query
    │
    ▼
┌─────────────┐     regex / keyword routing
│    Agent    │────────────────┬──────────────┬───────────────┐
└─────────────┘                │              │               │
       │ (paper question)      │ (math)       │ (keywords)    │ (word count)
       ▼                       ▼              ▼               ▼
┌─────────────┐         ┌────────────┐  ┌─────────────┐  ┌────────────┐
│  RAGEngine  │         │ Calculator │  │  Keyword    │  │    Word    │
│             │         │            │  │  Extractor  │  │  Counter   │
│ 1. embed    │         └────────────┘  └─────────────┘  └────────────┘
│ 2. FAISS    │
│    search   │
│ 3. Groq LLM │
│    (grounded│
│    answer)  │
└─────────────┘
```

## Project structure

```
rag-research-assistant/
├── app.py                      # Streamlit chat UI
├── src/
│   ├── rag_engine.py           # ingestion, chunking, embeddings, FAISS, generation
│   └── agent.py                # tool routing (calculator/keywords/word-count/RAG)
├── scripts/
│   └── download_papers.py      # reproducibly fetches the 8 papers from arXiv
├── data/papers/                 # downloaded PDFs (gitignored, regenerate locally)
├── requirements.txt
└── .streamlit/secrets.toml.example
```

## Setup

1. **Clone and install dependencies**

   ```bash
   git clone https://github.com/Adithya181/rag-research-assistant.git
   cd rag-research-assistant
   pip install -r requirements.txt
   ```

2. **Get a free Groq API key** at [console.groq.com/keys](https://console.groq.com/keys)

3. **Add your key locally** (never commit this file):

   ```bash
   mkdir -p .streamlit
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   # then edit .streamlit/secrets.toml and paste your real key
   ```

4. **Download the papers**

   ```bash
   python scripts/download_papers.py
   ```

5. **Run the app**

   ```bash
   streamlit run app.py
   ```

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing at your repo and `app.py`.
3. In the app's **Settings → Secrets**, add:

   ```toml
   GROQ_API_KEY = "your-real-key"
   ```

4. Note: `data/papers/` is gitignored, so add a one-time build step or run `scripts/download_papers.py` in the app's startup (e.g. via a small `if not os.path.exists(...)` check at the top of `app.py`) so Streamlit Cloud fetches the papers on first boot.

## Example queries

| Query                                                                      | Tool used          |
| --------------------------------------------------------------------------- | ------------------- |
| "What is the attention mechanism in transformers?"                          | RAG                 |
| "Explain how ResNet solves the vanishing gradient problem"                  | RAG                 |
| "calculate 50+6\*7%10-5"                                                     | Calculator          |
| "extract keywords from Artificial Intelligence is transforming industries"  | Keyword extractor   |
| "word count of The quick brown fox jumps over the lazy dog"                 | Word counter        |

## Tech stack

Python · Streamlit · Sentence-Transformers · FAISS · Groq (LLaMA 3.1) · pypdf

## Troubleshooting

- **`GROQ_API_KEY` not found** — make sure `.streamlit/secrets.toml` exists locally (or `GROQ_API_KEY` is set under Settings → Secrets on Streamlit Cloud) and that the key name matches exactly.
- **Empty or missing `data/papers/`** — run `python scripts/download_papers.py` before starting the app; the folder is gitignored on purpose.
- **FAISS index errors after adding papers** — delete any cached index files and re-run ingestion so the index matches the current paper set.
- **Slow first response** — the first query loads the embedding model and builds/reads the FAISS index; subsequent queries are faster.

## Contributing

Issues and pull requests are welcome. If you're adding a new tool to the agent (beyond calculator/keywords/word-count), please include a short description of its routing trigger and an example query in your PR.

## License

MIT
