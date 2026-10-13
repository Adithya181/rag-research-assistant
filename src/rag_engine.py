"""
RAG Engine — LangChain version.

Replaces manual PDF parsing, chunking, embedding, and FAISS handling
with LangChain's standardized components. Same behavior as before
(distance-filtered retrieval, grounded generation via Groq), but with
less boilerplate and free index persistence.
"""

import hashlib
import logging
import os
from pathlib import Path

from groq import APIError, RateLimitError
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ---- Config ----
PAPERS_DIR = "data/papers"
INDEX_DIR = "data/faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
# Minimum relevance score (0–1, higher = more similar) required to keep
# a retrieved chunk. Uses LangChain's normalized relevance score rather
# than raw FAISS L2 distance, so this threshold is on a predictable
# scale regardless of embedding dimensionality/normalization.
# Tuned via grid search (scripts/evaluate_retrieval.py --tune) over 50 golden
# questions: threshold=0.10 achieves hit-rate@4=98% / MRR=0.91 vs the old
# default of 0.30 which gave hit-rate@4=82% / MRR=0.81.
# The LangChain FAISS relevance scorer for normalized L2 embeddings returns
# scores on a [0, 1] scale where many true-positive chunks score below 0.30,
# so a lower floor captures the correct paper without admitting noise.
RELEVANCE_THRESHOLD = 0.1
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


def _papers_checksum(papers_dir: str) -> str:
    """SHA-256 over sorted filenames + file sizes in papers_dir.

    Used as a lightweight freshness check: if the set of PDFs changes
    (a file is added, removed, or replaced), the checksum changes and
    the index is rebuilt automatically.  We deliberately avoid hashing
    full file content to keep startup fast.
    """
    p = Path(papers_dir)
    if not p.exists():
        return ""
    entries = sorted(
        (f.name, f.stat().st_size) for f in p.iterdir() if f.suffix == ".pdf"
    )
    return hashlib.sha256(str(entries).encode()).hexdigest()


class RAGEngine:
    def __init__(self, groq_api_key: str, force_rebuild: bool = False):
        # normalize_embeddings=True puts vectors on the unit sphere, so
        # FAISS L2 distance becomes a well-behaved, bounded (0–2) value
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
        self.vectorstore = self._load_or_build_index(force_rebuild=force_rebuild)
        self.chain = self._build_chain()

    # ---------------------------------------------------------------
    # Index building / loading
    # ---------------------------------------------------------------
    def _load_or_build_index(self, force_rebuild: bool = False) -> FAISS:
        """Load a persisted FAISS index if it is still fresh, else
        (re)build it from the PDFs and save it for future runs.

        Freshness is determined by a lightweight checksum over the PDF
        filenames and sizes stored in a sidecar file next to the index.
        If the papers folder changes (file added / removed / swapped),
        the checksum mismatches and the index is rebuilt automatically.

        Security note: index.pkl is a pickle file produced by *this*
        codebase from PDFs we control.  allow_dangerous_deserialization
        is therefore intentional; never load an index from an untrusted
        source without verifying its provenance.
        """
        checksum_file = os.path.join(INDEX_DIR, ".papers_checksum")
        current_checksum = _papers_checksum(PAPERS_DIR)

        def _stored_checksum() -> str:
            try:
                return Path(checksum_file).read_text().strip()
            except FileNotFoundError:
                return ""

        index_exists = os.path.exists(INDEX_DIR) and os.path.exists(
            os.path.join(INDEX_DIR, "index.faiss")
        )
        checksum_matches = _stored_checksum() == current_checksum

        if index_exists and checksum_matches and not force_rebuild:
            logger.info("Loading existing FAISS index from %s", INDEX_DIR)
            return FAISS.load_local(
                INDEX_DIR, self.embeddings, allow_dangerous_deserialization=True
            )

        logger.info("Building FAISS index from PDFs in %s …", PAPERS_DIR)
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

        # Persist the checksum so the next run can skip rebuilding
        Path(checksum_file).write_text(current_checksum)
        logger.info("Index saved to %s", INDEX_DIR)

        return vectorstore

    # ---------------------------------------------------------------
    # Retrieval with relevance filtering
    # ---------------------------------------------------------------
    def _filtered_retrieve(self, query: str) -> list[Document]:
        """Retrieve top-k chunks, dropping anything below the
        relevance threshold so out-of-scope questions don't get
        forced context and hallucinated answers."""
        docs_and_scores = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=TOP_K
        )
        filtered = [doc for doc, score in docs_and_scores if score >= RELEVANCE_THRESHOLD]
        logger.debug(
            "Retrieved %d/%d chunks above threshold %.2f for query: %.60s",
            len(filtered),
            len(docs_and_scores),
            RELEVANCE_THRESHOLD,
            query,
        )
        return filtered

    # ---------------------------------------------------------------
    # Generation chain (LCEL)
    # ---------------------------------------------------------------
    def _build_chain(self):
        def format_docs(docs: list[Document]) -> str:
            if not docs:
                return "No relevant context found."
            return "\n\n".join(
                f"[{d.metadata.get('source', 'unknown')} p.{d.metadata.get('page', '?')}] {d.page_content}"
                for d in docs
            )

        # Build the chain so it takes pre-retrieved docs rather than
        # re-running retrieval.  query() retrieves once, passes the
        # docs here — no double embedding call.
        chain = (
            {
                "context": lambda x: format_docs(x["docs"]),
                "question": lambda x: x["question"],
            }
            | RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )
        return chain

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------
    def query(self, question: str) -> dict:
        """Retrieve once, generate once.

        Previously this called _filtered_retrieve() in the chain lambda
        *and* again below to build the sources list — doubling the
        embedding cost and risking source/context drift.  Now retrieval
        happens exactly once and the same docs are used for both.
        """
        docs = self._filtered_retrieve(question)

        try:
            answer = self.chain.invoke({"question": question, "docs": docs})
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
        seen: set[tuple] = set()
        sources = []
        for d in docs:
            key = (d.metadata.get("source", "unknown"), d.metadata.get("page"))
            if key not in seen:
                seen.add(key)
                sources.append({"source": key[0], "page": key[1]})

        return {"answer": answer, "sources": sources}