"""
Quick diagnostic — run this from your project root:
    python debug_retrieval.py

It checks:
  1. What files are in data/papers/
  2. What sources actually exist inside the persisted FAISS index
  3. The raw similarity scores for a "dropout" query, before filtering
"""

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

PAPERS_DIR = "data/papers"
INDEX_DIR = "data/faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

print("=" * 60)
print("1. Files in data/papers/")
print("=" * 60)
if os.path.exists(PAPERS_DIR):
    for f in os.listdir(PAPERS_DIR):
        print(" -", f)
else:
    print("  PAPERS_DIR does not exist!")

print()
print("=" * 60)
print("2. Sources present inside the persisted FAISS index")
print("=" * 60)
if os.path.exists(INDEX_DIR):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vs = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    sources = set()
    for doc_id in vs.index_to_docstore_id.values():
        doc = vs.docstore.search(doc_id)
        sources.add(doc.metadata.get("source", "unknown"))
    print(f"  {len(sources)} unique source(s) found in index:")
    for s in sorted(sources):
        print(" -", s)
else:
    print("  No persisted index found at", INDEX_DIR, "(will be built on first run)")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vs = None

print()
print("=" * 60)
print("3. Raw similarity scores for 'what is dropout' (before filtering)")
print("=" * 60)
if vs is not None:
    results = vs.similarity_search_with_score("what is dropout", k=6)
    if not results:
        print("  No results returned at all — index may be empty.")
    for doc, score in results:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"  score={score:.4f}  source={src} p.{page}  | {preview}...")
    print()
    print("  Current DISTANCE_THRESHOLD = 0.8")
    print("  Anything with score > 0.8 above is being FILTERED OUT.")
else:
    print("  Skipped — no index to query.")
