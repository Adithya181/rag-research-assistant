"""
Streamlit chat UI for the Multi-Tool RAG Research Assistant.
Uses the LangChain-based RAGEngine and tool-calling agent.
"""

import logging

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src.agent import build_agent, run_agent
from src.rag_engine import RAGEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title="RAG Research Assistant", page_icon="🤖", layout="wide")

# ---------------------------------------------------------------
# Per-session rate limiting
# Max requests per session to protect the shared Groq API key.
# Adjust MAX_REQUESTS_PER_SESSION to suit your quota.
# ---------------------------------------------------------------
MAX_REQUESTS_PER_SESSION = 30

# ---------------------------------------------------------------
# Friendly tool badges
# ---------------------------------------------------------------
TOOL_BADGES = {
    "knowledge_base_search": "🧠 Knowledge Base Search",
    "calculator": "🧮 Calculator",
    "extract_keywords": "🔑 Keyword Extractor",
    "word_count": "📝 Word Counter",
    "LLM": "💭 Direct Answer",
    "rate_limited": "⏳ Rate Limited",
    "api_error": "⚠️ API Error",
}


def format_tool_label(tool_string: str) -> str:
    """Map a (possibly chained) tool string like
    'calculator → word_count' to friendly badges."""
    parts = [p.strip() for p in tool_string.split("→")]
    labels = [TOOL_BADGES.get(p, p) for p in parts]
    return " → ".join(labels)


# ---------------------------------------------------------------
# Cached resources — only build the index / agent once per process
# ---------------------------------------------------------------
@st.cache_resource
def get_rag_engine() -> RAGEngine:
    return RAGEngine(groq_api_key=st.secrets["GROQ_API_KEY"])


@st.cache_resource
def get_agent(_rag_engine: RAGEngine) -> object:
    return build_agent(st.secrets["GROQ_API_KEY"], _rag_engine)


rag_engine = get_rag_engine()
agent_executor = get_agent(rag_engine)

# ---------------------------------------------------------------
# Session state initialisation — one clear place, before any widgets
# ---------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []          # {"role", "content", "tool"}
if "request_count" not in st.session_state:
    st.session_state.request_count = 0

# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
with st.sidebar:
    st.header("🤖 About")
    st.markdown(
        """
This assistant answers questions over 8 foundational ML papers
(Transformer, BERT, ResNet, GPT-3, Adam, Dropout, Batch Norm, ViT)
using Retrieval-Augmented Generation.

It also has a calculator, keyword extractor, and word counter —
the agent decides which tool fits your question.

**Try asking:**
- "What is the attention mechanism in transformers?"
- "calculate 50+6*7%10-5"
- "extract keywords from Artificial Intelligence is transforming industries"
- "word count of The quick brown fox jumps over the lazy dog"
        """
    )
    st.caption(
        f"Requests this session: {st.session_state.request_count} / {MAX_REQUESTS_PER_SESSION}"
    )
    if st.button(":material/delete: Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------
st.title("🤖 Multi-Tool RAG Research Assistant")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("tool"):
            st.caption(f"🔧 Tool used: {format_tool_label(msg['tool'])}")

# ---------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------
user_query = st.chat_input("Ask about the papers, or try the calculator/keyword tools...")

if user_query:
    # Soft per-session rate limit
    if st.session_state.request_count >= MAX_REQUESTS_PER_SESSION:
        st.warning(
            f"You've reached the session limit of {MAX_REQUESTS_PER_SESSION} requests. "
            "Refresh the page to start a new session."
        )
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Build LangChain message history for follow-up awareness
    lc_history = []
    for m in st.session_state.messages[:-1]:   # exclude the just-added user message
        if m["role"] == "user":
            lc_history.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_history.append(AIMessage(content=m["content"]))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = run_agent(agent_executor, user_query, chat_history=lc_history)
            answer = result["answer"]
            tool_used = result["tool"]

        st.markdown(answer)
        st.caption(f"🔧 Tool used: {format_tool_label(tool_used)}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "tool": tool_used}
    )
    st.session_state.request_count += 1