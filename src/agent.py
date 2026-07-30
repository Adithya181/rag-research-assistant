"""
Multi-tool agent — LangChain version.

Replaces regex/keyword routing with real LLM-driven tool selection
using LangChain's tool-calling agent. The LLM itself decides which
tool to call based on the user's query.
"""

import re
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.rag_engine import RAGEngine


def build_agent(groq_api_key: str, rag_engine: RAGEngine) -> AgentExecutor:
    # ---- Tools ----
    @tool
    def knowledge_base_search(query: str) -> str:
        """Answer questions about the 8 ML papers (Transformer, BERT,
        ResNet, GPT-3, Adam, Dropout, Batch Normalization, ViT) using
        retrieval-augmented generation over the paper text. Use this
        for any conceptual or technical question about the papers."""
        result = rag_engine.query(query)
        sources = ", ".join(s["source"] for s in result["sources"]) or "no matching source"
        return f"{result['answer']}\n\n(Sources: {sources})"

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a basic arithmetic expression, e.g. '50+6*7%10-5'.
        Only supports +, -, *, /, %, parentheses, and numbers."""
        if not re.fullmatch(r"[0-9\.\+\-\*\/\%\(\)\s]+", expression):
            return "Invalid expression — only numbers and + - * / % ( ) are allowed."
        try:
            return str(eval(expression, {"__builtins__": {}}))
        except Exception as e:
            return f"Could not evaluate expression: {e}"

    @tool
    def extract_keywords(text: str) -> str:
        """Extract the most significant keywords from a passage of text."""
        stopwords = {
            "is", "the", "a", "an", "and", "or", "of", "to", "in", "for",
            "on", "with", "are", "was", "were", "be", "by", "at", "from",
        }
        words = re.findall(r"[A-Za-z]+", text.lower())
        keywords = sorted(set(w for w in words if w not in stopwords and len(w) > 3))
        return ", ".join(keywords) if keywords else "No significant keywords found."

    @tool
    def word_count(text: str) -> str:
        """Count the number of words in a passage of text."""
        count = len(text.split())
        return f"Word count: {count}"

    tools = [knowledge_base_search, calculator, extract_keywords, word_count]

    # ---- Agent ----
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        groq_api_key=groq_api_key,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful research assistant. Decide which tool "
                "fits the user's request — the knowledge base for paper "
                "questions, the calculator for math, or the text utilities "
                "for keyword/word-count requests. Only use knowledge_base_search "
                "for questions actually about the 8 ML papers.",
            ),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    return executor


def run_agent(executor: AgentExecutor, query: str) -> dict:
    """Runs the agent and returns the answer plus which tool(s) fired,
    so app.py can still show the tool-type badge in the UI."""
    result = executor.invoke({"input": query})

    tool_used = "unknown"
    for step in result.get("intermediate_steps", []):
        action = step[0]
        tool_used = action.tool
        break

    return {"answer": result["output"], "tool": tool_used}
