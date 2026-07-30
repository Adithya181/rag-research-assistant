# 🤖 Multi-Tool RAG Research Assistant

🔗 **[Live Demo](https://rag-research-assistant-fuzjfaqbygqztvishbrv6c.streamlit.app)**

A Retrieval-Augmented Generation (RAG) system over 8 foundational machine
learning papers (Transformer, BERT, ResNet, GPT-3, Adam, Dropout, Batch
Normalization, ViT), wrapped in a LangChain-powered tool-calling agent and
served through an interactive Streamlit chat UI.

Ask it to explain a paper's architecture, run a calculation, extract
keywords, or count words in a passage — the agent decides which tool to
use and answers accordingly, citing sources for anything grounded in the
papers.

## Features

- **RAG pipeline (LangChain)**: PDF ingestion via `PyPDFDirectoryLoader` →
  chunking via `RecursiveCharacterTextSplitter` → `all-MiniLM-L6-v2`
  sentence embeddings (`HuggingFaceEmbeddings`) → FAISS similarity search →
  grounded answer generation via a Groq-hosted LLaMA 3.1 model, wired
  together with LCEL.
- **LLM-driven tool-calling agent**: built with LangChain's
  `create_tool_calling_agent` — the LLM itself decides which tool fits
  each query (RAG, calculator, keyword extractor, or word counter),
  rather than relying on keyword/regex matching.
- **Distance-filtered retrieval**: chunks beyond a similarity threshold
  are dropped instead of being forced into the prompt, reducing
  hallucinated "answers" to out-of-scope questions.
- **Persistent FAISS index**: the vector index is built once and saved to
  disk, so the app doesn't re-embed all papers on every restart.
- **Streamlit chat UI**: persistent chat history, source citations,
  tool-type badges, and a sidebar explaining what the agent can do.

## Architecture

```
User query
    │
    ▼
┌──────────────────────────┐
│ LangChain Tool-Calling   │  LLM decides which tool to call
│ Agent                    │
└──────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────┐
│ Tools                                      │
│ - knowledge_base_search (RAG)              │
│ - calculator                               │
│ - extract_keywords                         │
│ - word_count                               │
└───────────────────────────────────────────┘
    │
    ▼
knowledge_base_search → RAGEngine
    1. embed query (HuggingFaceEmbeddings)
    2. FAISS similarity search + distance filter
    3. Groq LLM generates grounded answer (LCEL chain)
    │
    ▼
Final answer returned to user, with tool badge shown in UI
```

## Project structure

```
rag-research-assistant/
├── app.py                      # Streamlit chat UI
├── src/
│   ├── rag_engine.py           # LangChain RAG: ingestion, chunking, embeddings, FAISS, generation
│   └── agent.py                # LangChain tool-calling agent (calculator/keywords/word-count/RAG)
├── scripts/
│   └── download_papers.py      # reproducibly fetches the 8 papers from arXiv
├── data/
│   ├── papers/                 # downloaded PDFs (gitignored, regenerate locally)
│   └── faiss_index/            # persisted vector index (gitignored, auto-built on first run)
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
   ```
   then edit `.streamlit/secrets.toml` and paste your real key

4. **Download the papers**
   ```bash
   python scripts/download_papers.py
   ```

5. **Run the app**
   ```bash
   streamlit run app.py
   ```

First run builds and saves the FAISS index (takes a bit longer);
subsequent runs load it instantly from `data/faiss_index/`.

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing at your repo and `app.py`.
3. In the app's **Settings → Secrets**, add:
   ```
   GROQ_API_KEY = "your-real-key"
   ```
4. Note: `data/papers/` and `data/faiss_index/` are gitignored, so add a
   one-time build step or run `scripts/download_papers.py` in the app's
   startup (e.g. via a small `if not os.path.exists(...)` check at the
   top of `app.py`) so Streamlit Cloud fetches the papers and builds the
   index on first boot.

## Example queries

| Query                                                                      | Tool used              |
| --------------------------------------------------------------------------| ----------------------- |
| "What is the attention mechanism in transformers?"                        | knowledge_base_search   |
| "Explain how ResNet solves the vanishing gradient problem"                | knowledge_base_search   |
| "calculate 50+6\*7%10-5"                                                  | calculator              |
| "extract keywords from Artificial Intelligence is transforming industries"| extract_keywords        |
| "word count of The quick brown fox jumps over the lazy dog"               | word_count               |

## Tech stack

Python · Streamlit · LangChain · Sentence-Transformers · FAISS · Groq (LLaMA 3.1) · pypdf

## License

MIT
