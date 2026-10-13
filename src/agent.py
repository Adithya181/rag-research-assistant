"""
Multi-tool agent — LangChain version.

Replaces regex/keyword routing with real LLM-driven tool selection
using LangChain's tool-calling agent. The LLM itself decides which
tool to call based on the user's query.
"""

import ast
import logging
import operator
import re
from collections import deque
from typing import Any

from groq import APIError, RateLimitError
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from src.rag_engine import RAGEngine

logger = logging.getLogger(__name__)

# ---- Safe arithmetic evaluator (replaces eval()) ----
_ALLOWED_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Cap exponents to prevent 9**9**9**9 style denial-of-service
_MAX_EXPONENT = 1_000


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate a parsed AST, allowing only safe arithmetic."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if op_type is ast.Pow and abs(right) > _MAX_EXPONENT:
            raise ValueError(
                f"Exponent {right} exceeds the safety limit ({_MAX_EXPONENT})."
            )
        return _ALLOWED_OPS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return _ALLOWED_OPS[op_type](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression type: {type(node).__name__}")


def safe_calculate(expression: str) -> str:
    """Parse and evaluate expression via AST — no eval(), no DoS risk."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as e:
        return f"Invalid expression syntax: {e}"
    try:
        result = _safe_eval(tree)
    except (ValueError, ZeroDivisionError) as e:
        return f"Could not evaluate: {e}"
    # Return int if result is a whole number
    if isinstance(result, float) and result.is_integer():
        return str(int(result))
    return str(result)


def build_agent(groq_api_key: str, rag_engine: RAGEngine) -> AgentExecutor:
    # ---- Tools ----
    @tool
    def knowledge_base_search(query: str) -> str:
        """Answer questions about the 8 ML papers (Transformer, BERT,
        ResNet, GPT-3, Adam, Dropout, Batch Normalization, ViT) using
        retrieval-augmented generation over the paper text. Use this
        for any conceptual or technical question about the papers."""
        result = rag_engine.query(query)
        sources = (
            ", ".join(s["source"] for s in result["sources"]) or "no matching source"
        )
        return f"{result['answer']}\n\n(Sources: {sources})"

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a basic arithmetic expression safely.

        Supports +, -, *, /, %, ** (capped at exponent ≤ 1000), and
        parentheses.  Uses an AST-based evaluator — no eval(), no
        risk of code injection or denial-of-service via 9**9**9**9.

        Examples: '50+6*7%10-5', '(3+4)*2', '2**10'
        """
        return safe_calculate(expression)

    @tool
    def extract_keywords(text: str) -> str:
        """Extract statistically significant keywords from a passage of
        text using TF-style term frequency, filtered by a stopword list.

        Returns up to 10 keywords ranked by frequency (most frequent
        first), which is more useful than a flat alphabetical dump.
        """
        stopwords = {
            "is", "the", "a", "an", "and", "or", "of", "to", "in", "for",
            "on", "with", "are", "was", "were", "be", "by", "at", "from",
            "that", "this", "it", "its", "as", "has", "have", "had",
            "not", "but", "so", "if", "can", "also", "which", "their",
            "they", "we", "our", "you", "your", "i", "me", "my",
        }
        words = re.findall(r"[A-Za-z]+", text.lower())
        freq: dict[str, int] = {}
        for w in words:
            if w not in stopwords and len(w) > 3:
                freq[w] = freq.get(w, 0) + 1
        if not freq:
            return "No significant keywords found."
        ranked = sorted(freq.items(), key=lambda x: -x[1])[:10]
        return ", ".join(f"{w} ({c})" for w, c in ranked)

    @tool
    def word_count(text: str) -> str:
        """Count words, sentences, and characters in a passage of text.

        Returns a breakdown rather than just a single number, which is
        more informative for document analysis tasks.
        """
        word_list = text.split()
        sentence_count = max(
            1, len(re.findall(r"[.!?]+", text)) or 1
        )
        char_count = len(text)
        char_no_space = len(text.replace(" ", ""))
        return (
            f"Words: {len(word_list)} | Sentences: {sentence_count} | "
            f"Characters: {char_count} (excluding spaces: {char_no_space})"
        )

    tools = [knowledge_base_search, calculator, extract_keywords, word_count]

    # ---- Agent ----
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=groq_api_key,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an intelligent research assistant. You have
EXACTLY four tools available, and no others: knowledge_base_search,
calculator, extract_keywords, word_count.

You do NOT have access to the internet, a web browser, or any other
tool of any kind — including but not limited to brave_search,
web_search, wolfram_alpha, or code_interpreter. These tools do not
exist in this environment. NEVER attempt to call them under any
circumstances.

Classify every user message into exactly one of these categories,
and respond accordingly:

1. ABOUT THE 8 PAPERS
   The question is about, mentions, or asks to explain/define ANY of
   these 8 topics, in any phrasing — including simple "what is X" or
   "explain X" questions: Transformer, BERT, ResNet, GPT-3, Adam,
   Batch Normalization, Dropout, Vision Transformer (ViT).
   → Call the knowledge_base_search tool. Do NOT answer from your own
   knowledge instead, even if you already know the answer.
   Examples that MUST use this tool:
     "what is dropout" → knowledge_base_search
     "what is batch normalization" → knowledge_base_search
     "explain the attention mechanism" → knowledge_base_search
     "how does ResNet work" → knowledge_base_search

2. MATH / KEYWORDS / WORD COUNT
   → Call calculator, extract_keywords, or word_count as appropriate.

3. GENERAL ML / AI / COMPUTER SCIENCE KNOWLEDGE
   The question is about machine learning, AI, or related CS concepts,
   but is NOT specifically about the 8 papers above (e.g. "what is
   overfitting", "explain gradient descent", "what's the difference
   between CNNs and RNNs").
   → Do NOT call any tool. Answer directly and helpfully from your own
   knowledge.

4. COMPLETELY UNRELATED
   The question has nothing to do with ML, AI, or CS (e.g. "what's
   the capital of France", "tell me a joke", "what's the weather").
   → Do NOT call any tool. Politely decline: say that you can only
   answer questions about the 8 ML papers, general ML/AI concepts, or
   use the calculator, keyword extractor, and word counter tools.

Always pick exactly one category. Never blend categories and never
call a tool that doesn't exist.""",
            ),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,          # Don't leak internal traces to stdout/logs
        return_intermediate_steps=True,
        max_iterations=6,       # Prevent runaway loops from burning quota
        max_execution_time=30,  # Hard wall-clock timeout (seconds)
    )

    return executor


def run_agent(
    executor: AgentExecutor,
    query: str,
    chat_history: list[BaseMessage] | None = None,
) -> dict:
    """Run the agent and return answer + tool(s) used.

    Args:
        executor: The AgentExecutor built by build_agent().
        query: The current user message.
        chat_history: LangChain message objects from prior turns so
            follow-up questions like "explain that more simply" work.

    Returns:
        {"answer": str, "tool": str}
    """
    payload: dict[str, Any] = {
        "input": query,
        "chat_history": chat_history or [],
    }

    try:
        result = executor.invoke(payload)
    except RateLimitError:
        return {
            "answer": (
                "I'm getting rate-limited by the Groq API right now "
                "(too many requests in a short window, or the daily/"
                "per-minute quota on this API key has been reached). "
                "Please wait a minute and try again. If this keeps "
                "happening, check your usage at "
                "console.groq.com/settings/limits."
            ),
            "tool": "rate_limited",
        }
    except APIError as e:
        return {
            "answer": (
                f"The Groq API returned an error and couldn't complete "
                f"this request ({e.__class__.__name__}). Please try "
                f"again in a moment."
            ),
            "tool": "api_error",
        }

    intermediate_steps = result.get("intermediate_steps", [])
    output = result["output"]

    if intermediate_steps:
        tools_used = [action.tool for action, _obs in intermediate_steps]
        # Deduplicate while preserving order
        tool_used = " → ".join(dict.fromkeys(tools_used))
    else:
        # No tool called — LLM answered directly (categories 3 or 4)
        tool_used = "LLM"

    logger.info("Query='%.60s' tool=%s", query, tool_used)
    return {"answer": output, "tool": tool_used}
