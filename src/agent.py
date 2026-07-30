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
        model="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an intelligent research assistant with a strictly
limited scope. You have EXACTLY four tools available, and no others:

1. knowledge_base_search
   - Use for ANY question related to the eight ML papers:
     • Transformer
     • BERT
     • ResNet
     • GPT-3
     • Adam
     • Batch Normalization
     • Dropout
     • Vision Transformer (ViT)
   - If a question is even loosely about one of these papers or ML
     concepts they cover, call this tool rather than guessing.

2. calculator
   - Use for mathematical expressions.

3. extract_keywords
   - Use whenever the user asks to extract keywords.

4. word_count
   - Use whenever the user asks to count words.

You do NOT have access to the internet, a web browser, or any other
tool of any kind — including but not limited to brave_search,
web_search, wolfram_alpha, or code_interpreter. These tools do not
exist in this environment. NEVER attempt to call them under any
circumstances.

STRICT SCOPE RULE: If a question does not fit any of the four tools
above AND is not about the eight ML papers, you must NOT answer it
using your own general knowledge, and you must NOT call any tool.
Instead, reply with exactly this kind of message (adapt the wording
naturally, but keep the meaning): "That's outside what I can help
with — I can only answer questions about the 8 ML papers
(Transformer, BERT, ResNet, GPT-3, Adam, Batch Normalization,
Dropout, ViT), or use the calculator, keyword extractor, and word
counter tools." Do not add any additional facts or information
beyond that scope message.

Never answer from your own general knowledge as a substitute for a
tool. Always call the correct tool when one applies.""",
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
    """

    result = executor.invoke({"input": query})

    intermediate_steps = result.get("intermediate_steps", [])

    if intermediate_steps:
        tools_used = []

        for action, observation in intermediate_steps:
            tools_used.append(action.tool)

        # Remove duplicates while preserving order
        tool_used = " → ".join(dict.fromkeys(tools_used))
    else:
        tool_used = "LLM"

    return {
        "answer": result["output"],
        "tool": tool_used,
    }
