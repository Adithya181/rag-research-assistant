"""
Multi-tool agent — LangChain version.

Replaces regex/keyword routing with real LLM-driven tool selection
using LangChain's tool-calling agent. The LLM itself decides which
tool to call based on the user's query.
"""

import re
from groq import RateLimitError, APIError
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
   The question is directly about one of: Transformer, BERT, ResNet,
   GPT-3, Adam, Batch Normalization, Dropout, Vision Transformer (ViT).
   → Call the knowledge_base_search tool. Do not answer from your own
   knowledge instead.

2. MATH / KEYWORDS / WORD COUNT
   → Call calculator, extract_keywords, or word_count as appropriate.

3. GENERAL ML / AI / COMPUTER SCIENCE KNOWLEDGE
   The question is about machine learning, AI, or related CS concepts,
   but is NOT specifically about the 8 papers above (e.g. "what is
   overfitting", "explain gradient descent", "what's the difference
   between CNNs and RNNs").
   → Do NOT call any tool. Answer directly and helpfully from your own
   knowledge, but your response MUST start with the exact literal tag
   "[GENERAL_KNOWLEDGE]" followed by a space, then your answer.

4. COMPLETELY UNRELATED
   The question has nothing to do with ML, AI, or CS (e.g. "what's
   the capital of France", "tell me a joke", "what's the weather").
   → Do NOT call any tool. Do NOT answer the question at all. Your
   entire response MUST be exactly this literal tag followed by a
   space and this exact message:
   "[OUT_OF_SCOPE] That's outside what I can help with — I can only
   answer questions about the 8 ML papers (Transformer, BERT, ResNet,
   GPT-3, Adam, Batch Normalization, Dropout, ViT), general ML/AI
   concepts, or use the calculator, keyword extractor, and word
   counter tools."

Always pick exactly one category. Never blend categories, never skip
the required tag for categories 3 and 4, and never call a tool that
doesn't exist.""",
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
        verbose=True,
        return_intermediate_steps=True,
    )

    return executor


def run_agent(executor: AgentExecutor, query: str) -> dict:
    """
    Runs the agent and returns:
      - answer
      - tool(s) used
    Handles Groq rate limits and API errors gracefully instead of
    letting them crash the Streamlit app.
    """

    try:
        result = executor.invoke({"input": query})
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
        tools_used = []

        for action, observation in intermediate_steps:
            tools_used.append(action.tool)

        # Remove duplicates while preserving order
        tool_used = " → ".join(dict.fromkeys(tools_used))
    else:
        # No tool was called — check whether the LLM tagged this as
        # general knowledge or out-of-scope, per the system prompt.
        stripped = output.strip()
        if stripped.startswith("[OUT_OF_SCOPE]"):
            tool_used = "out_of_scope"
            output = stripped[len("[OUT_OF_SCOPE]"):].strip()
        elif stripped.startswith("[GENERAL_KNOWLEDGE]"):
            tool_used = "general_knowledge"
            output = stripped[len("[GENERAL_KNOWLEDGE]"):].strip()
        else:
            # Fallback: model didn't tag as instructed
            tool_used = "LLM"

    return {
        "answer": output,
        "tool": tool_used,
    }