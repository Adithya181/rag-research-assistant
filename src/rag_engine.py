"""
Core RAG engine: PDF ingestion, chunking, embeddings, FAISS retrieval,
and Groq-based answer generation.

This module is imported by both app.py (Streamlit UI) and any
notebook/script use, so the pipeline logic lives in exactly one place.
"""
import os
import glob
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping word chunks so a sentence sitting at a
    chunk boundary still appears fully in at least one chunk."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def load_and_chunk_papers(papers_dir: str = "data/papers") -> tuple[list[str], list[dict]]:
    """Read every PDF in papers_dir, extract text, and chunk it.
    Returns (all_chunks, chunk_metadata) where metadata tracks source file."""
    all_chunks, chunk_metadata = [], []
    for path in sorted(glob.glob(os.path.join(papers_dir, "*.pdf"))):
        reader = PdfReader(path)
        full_text = " ".join(page.extract_text() or "" for page in reader.pages)
        for chunk in chunk_text(full_text):
            all_chunks.append(chunk)
            chunk_metadata.append({"source": os.path.basename(path)})
    return all_chunks, chunk_metadata


class RAGEngine:
    """Wraps embedder + FAISS index + Groq client into one reusable object."""

    def __init__(self, groq_api_key: str, papers_dir: str = "data/papers",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        self.client = Groq(api_key=groq_api_key)
        self.embedder = SentenceTransformer(embedding_model)

        self.all_chunks, self.chunk_metadata = load_and_chunk_papers(papers_dir)
        if not self.all_chunks:
            raise RuntimeError(
                f"No PDFs found in '{papers_dir}'. Run scripts/download_papers.py first."
            )

        embeddings = self.embedder.encode(self.all_chunks, show_progress_bar=False)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings))

    def retrieve(self, query: str, top_k: int = 3, max_distance: float = 1.2) -> list[dict]:
        """Return top_k chunks within max_distance of the query embedding.
        The distance cutoff keeps clearly irrelevant chunks out of the prompt."""
        query_vector = self.embedder.encode([query])
        distances, indices = self.index.search(np.array(query_vector), top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if dist <= max_distance:
                results.append({
                    "text": self.all_chunks[idx],
                    "source": self.chunk_metadata[idx]["source"],
                    "distance": float(dist),
                })
        return results

    def generate_answer(self, query: str, top_k: int = 3,
                         model: str = "llama-3.1-8b-instant") -> dict:
        retrieved = self.retrieve(query, top_k=top_k)

        if not retrieved:
            return {
                "answer": "I couldn't find anything relevant to that in the indexed papers.",
                "sources": [],
            }

        context = "\n\n".join(
            f"[Source: {r['source']}]\n{r['text']}" for r in retrieved
        )
        prompt = f"""You are a research assistant answering questions about machine learning papers.
Use ONLY the context below to answer the question. If the context doesn't contain
enough information to answer, say so honestly instead of guessing.

Context:
{context}

Question:
{query}

Answer:
"""
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300,
        )
        return {
            "answer": response.choices[0].message.content,
            "sources": sorted(set(r["source"] for r in retrieved)),
        }
