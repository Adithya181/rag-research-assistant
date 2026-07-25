"""
Multi-tool agent: routes an incoming query to the right tool
(calculator, keyword extractor, word counter, or RAG) and returns
a structured, schema-validated response.
"""
import re

STOPWORDS = {"about", "which", "there", "their", "would", "these", "those",
             "extract", "keywords", "from"}

PAPER_KEYWORDS = [
    "paper", "explain", "what is", "how does", "describe", "architecture",
    "model", "transformer", "resnet", "bert", "attention", "optimizer",
    "dropout", "batch norm", "vit",
]


def calculator(expression: str) -> str:
    """Safely evaluate a basic arithmetic expression (+, -, *, /, %)."""
    try:
        cleaned = re.sub(r"[^0-9+\-*/%.() ]", "", expression).strip()
        if not cleaned:
            return "Error in calculation"
        return str(eval(cleaned, {"__builtins__": {}}, {}))
    except Exception:
        return "Error in calculation"


def looks_like_math(query: str) -> bool:
    stripped = query.strip().strip('"').strip("'")
    return bool(re.fullmatch(r"[0-9+\-*/%.() ]+", stripped)) and stripped != ""


def extract_keywords(text: str) -> list[str]:
    """Return up to 5 unique keywords longer than 4 characters."""
    try:
        words = [w.lower().strip(".,!?") for w in text.split() if len(w) > 4]
        words = [w for w in words if w not in STOPWORDS]
        seen, unique = set(), []
        for w in words:
            if w not in seen:
                seen.add(w)
                unique.append(w)
        return unique[:5]
    except Exception:
        return []


def word_counter(text: str) -> int:
    try:
        return len(text.split())
    except Exception:
        return 0


def looks_like_paper_question(query: str) -> bool:
    query_lower = query.lower()
    return any(kw in query_lower for kw in PAPER_KEYWORDS)


def validate_schema(response: dict) -> bool:
    return isinstance(response, dict) and "type" in response and "result" in response


class Agent:
    """Routes queries to calculator / keyword / word-count / RAG tools."""

    def __init__(self, rag_engine):
        self.rag_engine = rag_engine
        self.execution_log: list[dict] = []

    def run(self, query: str) -> dict:
        query_lower = query.lower()
        node = None
        try:
            if "calculate" in query_lower or looks_like_math(query):
                node = "calculator_tool"
                expr = query_lower.replace("calculate", "").strip().strip('"').strip("'")
                result = calculator(expr)
                response = ({"type": "error", "result": result}
                            if result == "Error in calculation"
                            else {"type": "calculation", "result": result})

            elif "keywords" in query_lower:
                node = "keyword_tool"
                text = query_lower.replace("extract keywords from", "").strip()
                response = {"type": "keywords", "result": extract_keywords(text)}

            elif "count words" in query_lower or "word count" in query_lower:
                node = "word_counter_tool"
                text = (query_lower.replace("count words in", "")
                                    .replace("word count of", "").strip())
                response = {"type": "word_count", "result": word_counter(text)}

            elif looks_like_paper_question(query):
                node = "rag_tool"
                rag_result = self.rag_engine.generate_answer(query)
                response = {
                    "type": "rag_answer",
                    "result": rag_result["answer"],
                    "sources": rag_result["sources"],
                }

            else:
                node = "general_response"
                response = {
                    "type": "general",
                    "result": f"You asked: '{query}'. This is a general query with no specific tool.",
                }

        except Exception as e:
            node = "error_handler"
            response = {"type": "error", "result": f"Something went wrong: {e}"}

        self.execution_log.append({"query": query, "node": node})

        if not validate_schema(response):
            response = {"type": "error", "result": "Response did not match expected schema"}

        return response
