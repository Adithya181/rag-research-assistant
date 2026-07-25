"""
Streamlit UI for the Multi-Tool RAG Research Assistant.

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    Set GROQ_API_KEY in the app's "Secrets" panel (see README.md).
"""
import os
import glob
import streamlit as st
from src.rag_engine import RAGEngine
from src.agent import Agent
from scripts.download_papers import download_papers

st.set_page_config(page_title="ML Papers Agent", page_icon="🤖", layout="centered")

# ---------- Auto-download papers on first boot (e.g. Streamlit Cloud) ----------
# data/papers/ is gitignored, so a freshly cloned deployment starts empty.
if not glob.glob("data/papers/*.pdf"):
    with st.spinner("First-time setup: downloading papers..."):
        download_papers("data/papers")

# ---------- API key: Streamlit secrets first, then env var (for local runs) ----------
def get_groq_api_key() -> str | None:
    try:
        return st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        return os.environ.get("GROQ_API_KEY")


groq_api_key = get_groq_api_key()

if not groq_api_key:
    st.error(
        "No GROQ_API_KEY found. Add it to `.streamlit/secrets.toml` locally, "
        "or in your Streamlit Cloud app's Secrets panel. See README.md."
    )
    st.stop()


# ---------- Build the pipeline once, cache across reruns ----------
@st.cache_resource(show_spinner="Indexing papers (first load only)...")
def load_engine():
    engine = RAGEngine(groq_api_key=groq_api_key, papers_dir="data/papers")
    return engine, Agent(engine)


try:
    rag_engine, agent = load_engine()
except RuntimeError as e:
    st.error(str(e))
    st.info("Run `python scripts/download_papers.py` first, then reload.")
    st.stop()


# ---------- Sidebar ----------
with st.sidebar:
    st.header("🤖 About this agent")
    st.markdown(
        "A **Retrieval-Augmented Generation** assistant over 8 classic ML "
        "papers, wrapped in a multi-tool agent."
    )
    st.markdown("**Papers indexed:**")
    st.markdown(
        "- Attention Is All You Need\n"
        "- BERT\n- ResNet\n- GPT-3\n- Adam Optimizer\n"
        "- Dropout\n- Batch Normalization\n- ViT"
    )
    st.markdown("**Tools available:**")
    st.markdown(
        "- 📄 **RAG** — ask about any of the papers above\n"
        "- 🧮 **Calculator** — e.g. `50+6*7%10`\n"
        "- 🔑 **Keyword extractor** — `extract keywords from ...`\n"
        "- 🔢 **Word counter** — `word count of ...`"
    )
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

st.title("🤖 ML Papers Multi-Tool Agent")
st.caption("Ask about 8 classic ML papers, or use the calculator / keyword / word-count tools.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("📚 Sources: " + ", ".join(msg["sources"]))

if prompt := st.chat_input("Ask a question, or try 'calculate 12*7'..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = agent.run(prompt)

        badge = {
            "rag_answer": "📄 RAG",
            "calculation": "🧮 Calculator",
            "keywords": "🔑 Keywords",
            "word_count": "🔢 Word count",
            "general": "💬 General",
            "error": "⚠️ Error",
        }.get(result["type"], result["type"])

        st.markdown(f"`{badge}`")
        st.markdown(result["result"])

        sources = result.get("sources")
        if sources:
            st.caption("📚 Sources: " + ", ".join(sources))

    st.session_state.messages.append({
        "role": "assistant",
        "content": f"`{badge}`\n\n{result['result']}",
        "sources": sources,
    })
