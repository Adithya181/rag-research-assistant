"""
Quick diagnostic — run this from your project root:
    python scripts/debug_retrieval.py

Checks:
  1. What PDFs are in data/papers/
  2. What sources the persisted FAISS index actually contains
  3. Raw similarity scores for a sample query (before filtering)

Uses the same embedding settings as rag_engine.py (normalized), so
scores here match what the live app sees.
"""

import os
import sys

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

PAPERS_DIR = "data/papers"
INDEX_DIR = "data/faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Must match rag_engine.RELEVANCE_THRESHOLD
RELEVANCE_THRESHOLD = 0.3
SAMPLE_QUERY = "what is dropout"


def main() -> None:
    sep = "=" * 60

    # 1. PDFs on disk
    print(sep)
    print("1. PDFs in", PAPERS_DIR)
    print(sep)
    if os.path.exists(PAPERS_DIR):
        pdfs = [f for f in os.listdir(PAPERS_DIR) if f.endswith(".pdf")]
        for f in sorted(pdfs):
            print(" -", f)
        if not pdfs:
            print("  (no PDFs found)")
    else:
        print("  PAPERS_DIR does not exist! Run scripts/download_papers.py first.")

    # 2. Index contents
    print()
    print(sep)
    print("2. Sources inside the persisted FAISS index")
    print(sep)

    vs = None
    if not os.path.exists(INDEX_DIR):
        print("  No index found at", INDEX_DIR, "— will be built on first app run.")
    else:
        # normalize_embeddings=True must match how the index was built
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
        vs = FAISS.load_local(
            INDEX_DIR, embeddings, allow_dangerous_deserialization=True
        )
        sources: set[str] = set()
        for doc_id in vs.index_to_docstore_id.values():
            doc = vs.docstore.search(doc_id)
            sources.add(doc.metadata.get("source", "unknown"))
        print(f"  {len(sources)} unique source(s) in index:")
        for s in sorted(sources):
            print(" -", s)

    # 3. Similarity scores for sample query
    print()
    print(sep)
    print(f"3. Relevance scores for '{SAMPLE_QUERY}' (threshold = {RELEVANCE_THRESHOLD})")
    print(sep)
    if vs is None:
        print("  Skipped — no index to query.")
    else:
        results = vs.similarity_search_with_relevance_scores(SAMPLE_QUERY, k=6)
        if not results:
            print("  No results — index may be empty.")
        for doc, score in results:
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            preview = doc.page_content[:80].replace("\n", " ")
            kept = "✓ kept" if score >= RELEVANCE_THRESHOLD else "✗ filtered"
            print(f"  [{kept}] score={score:.4f}  {src} p.{page} | {preview}…")


if __name__ == "__main__":
    # Make sure imports resolve when run from project root
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
