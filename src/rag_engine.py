"""
RAG Engine — LangChain version.

Replaces manual PDF parsing, chunking, embedding, and FAISS handling
with LangChain's standardized components. Same behavior as before
(distance-filtered retrieval, grounded generation via Groq), but with
less boilerplate and free index persistence.
"""

import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from groq import RateLimitError, APIError

# ---- Config ----
PAPERS_DIR = "data/papers"
INDEX_DIR = "data/faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
# Minimum relevance score (0-1, higher = more similar) required to keep
# a retrieved chunk. Uses LangChain's normalized relevance score rather
# than raw FAISS L2 distance, so this threshold is on a predictable
# scale regardless of embedding dimensionality/normalization.
RELEVANCE_THRESHOLD = 0.3
TOP_K = 4

RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a research assistant answering questions about machine
learning papers. Use ONLY the context below to answer. If the context
doesn't contain the answer, say you don't have enough information —
do not make anything up.

Context:
{context}

Question: {question}

Answer, citing which paper(s) you drew from where relevant:"""
)


class RAGEngine:
    def __init__(self, groq_api_key: str):
        # normalize_embeddings=True puts vectors on the unit sphere, so
        # FAISS L2 distance becomes a well-behaved, bounded (0-2) value
        # and cosine-similarity-based relevance scoring works correctly.
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
        self.llm = ChatGroq(
            model="openai/gpt-oss-120b",
            groq_api_key=groq_api_key,
            temperature=0,
        )
        self.vectorstore = self._load_or_build_index()
        self.chain = self._build_chain()

    # ---------------------------------------------------------------
    # Index building / loading
    # ---------------------------------------------------------------
    def _load_or_build_index(self) -> FAISS:
        """Load a persisted FAISS index if it exists, else build one
        from the PDFs and save it so future runs skip re-embedding."""
        if os.path.exists(INDEX_DIR):
            return FAISS.load_local(
                INDEX_DIR, self.embeddings, allow_dangerous_deserialization=True
            )

        loader = PyPDFDirectoryLoader(PAPERS_DIR)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        vectorstore.save_local(INDEX_DIR)
        return vectorstore

    # ---------------------------------------------------------------
    # Retrieval with relevance filtering
    # ---------------------------------------------------------------
    def _filtered_retrieve(self, query: str):
        """Retrieve top-k chunks, dropping anything below the
        relevance threshold so out-of-scope questions don't get
        forced context and hallucinated answers."""
        docs_and_scores = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=TOP_K
        )

        filtered_docs = [
            doc for doc, score in docs_and_scores if score >= RELEVANCE_THRESHOLD
        ]

        return filtered_docs

    # ---------------------------------------------------------------
    # Generation chain (LCEL)
    # ---------------------------------------------------------------
    def _build_chain(self):
        def format_docs(docs):
            if not docs:
                return "No relevant context found."
            return "\n\n".join(
                f"[{d.metadata.get('source', 'unknown')} p.{d.metadata.get('page', '?')}] {d.page_content}"
                for d in docs
            )

        chain = (
            {
                "context": lambda x: format_docs(self._filtered_retrieve(x["question"])),
                "question": lambda x: x["question"],
            }
            | RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )
        return chain

    # ---------------------------------------------------------------
    # Public API — keep this signature close to your original so
    # app.py / agent.py need minimal changes.
    # ---------------------------------------------------------------
    def query(self, question: str) -> dict:
        docs = self._filtered_retrieve(question)

        try:
            answer = self.chain.invoke({"question": question})
        except RateLimitError:
            return {
                "answer": (
                    "I'm getting rate-limited by the Groq API right now. "
                    "Please wait a minute and try again."
                ),
                "sources": [],
            }
        except APIError:
            return {
                "answer": (
                    "The Groq API returned an error and couldn't "
                    "complete this request. Please try again shortly."
                ),
                "sources": [],
            }

        # Deduplicate sources while preserving order
        seen = set()
        sources = []
        for d in docs:
            key = (d.metadata.get("source", "unknown"), d.metadata.get("page"))
            if key not in seen:
                seen.add(key)
                sources.append({"source": key[0], "page": key[1]})

        return {"answer": answer, "sources": sources}