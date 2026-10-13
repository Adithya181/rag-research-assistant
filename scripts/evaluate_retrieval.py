"""
Retrieval evaluation script.

Measures how reliably the RAG pipeline retrieves the correct source
paper for each question in the golden set.

Metrics computed
----------------
hit@k   True if the expected source appears in the top-k retrieved
        chunks.  We report hit@1, hit@2, and hit@4 (TOP_K).

        hit-rate@k = (# questions where the expected source was in
                      top-k results) / total questions

MRR     Mean Reciprocal Rank — average of 1/rank of the first hit.
        Closer to 1.0 is better; accounts for whether the right paper
        appears first vs last.

Usage
-----
    python scripts/evaluate_retrieval.py

Optional flags
--------------
    --golden tests/golden_set.json   path to the golden set
    --k 4                            max chunks to retrieve (must match TOP_K)
    --threshold 0.3                  relevance threshold to evaluate
    --tune                           grid-search threshold & chunk settings,
                                     print a comparison table
    --out results/eval_TIMESTAMP.json  save raw results to JSON
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Make sure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.rag_engine import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    INDEX_DIR,
    RELEVANCE_THRESHOLD,
    TOP_K,
)


# ---------------------------------------------------------------------------
# Core evaluation helpers
# ---------------------------------------------------------------------------

def load_golden_set(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def build_vectorstore(embedding_model: str = EMBEDDING_MODEL) -> FAISS:
    """Load the committed FAISS index with normalized embeddings."""
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.load_local(
        INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )


def retrieve(
    vectorstore: FAISS,
    question: str,
    k: int = TOP_K,
    threshold: float = RELEVANCE_THRESHOLD,
) -> list[dict]:
    """Return filtered chunks as {source, page, score} dicts."""
    results = vectorstore.similarity_search_with_relevance_scores(question, k=k)
    return [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "?"),
            "score": round(score, 4),
        }
        for doc, score in results
        if score >= threshold
    ]


def evaluate(
    golden: list[dict],
    vectorstore: FAISS,
    k: int = TOP_K,
    threshold: float = RELEVANCE_THRESHOLD,
) -> dict:
    """Run retrieval for every golden question and compute metrics."""
    hits_at = {1: 0, 2: 0, k: 0}
    reciprocal_ranks = []
    details = []

    for item in golden:
        question = item["question"]
        expected = item["expected_source"]
        chunks = retrieve(vectorstore, question, k=k, threshold=threshold)
        sources = [c["source"] for c in chunks]

        # Rank of first hit (1-indexed); None if not found
        rank = next((i + 1 for i, s in enumerate(sources) if s == expected), None)

        hit1 = rank == 1
        hit2 = rank is not None and rank <= 2
        hitk = rank is not None

        hits_at[1] += int(hit1)
        hits_at[2] += int(hit2)
        hits_at[k] += int(hitk)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        details.append(
            {
                "id": item["id"],
                "question": question,
                "expected": expected,
                "rank": rank,
                "hit@1": hit1,
                "hit@2": hit2,
                f"hit@{k}": hitk,
                "retrieved_sources": sources,
                "chunks": chunks,
            }
        )

    n = len(golden)
    metrics = {
        "n_questions": n,
        "k": k,
        "threshold": threshold,
        "hit_rate@1": round(hits_at[1] / n, 4),
        "hit_rate@2": round(hits_at[2] / n, 4),
        f"hit_rate@{k}": round(hits_at[k] / n, 4),
        "mrr": round(sum(reciprocal_ranks) / n, 4),
        "misses": [d for d in details if not d[f"hit@{k}"]],
    }
    return metrics, details


# ---------------------------------------------------------------------------
# Grid search for tuning
# ---------------------------------------------------------------------------

def tune(golden: list[dict]) -> None:
    """Try a grid of threshold values with the committed index and
    print a comparison table so you can pick the best setting."""

    thresholds = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    ks = [1, 2, 4]

    print("\n" + "=" * 72)
    print(f"{'threshold':>10}  {'k':>3}  {'hit@1':>7}  {'hit@2':>7}  {'hit@k':>7}  {'mrr':>7}")
    print("-" * 72)

    vs = build_vectorstore()
    best = None

    for thr in thresholds:
        for k in ks:
            metrics, _ = evaluate(golden, vs, k=k, threshold=thr)
            row = (
                f"{thr:>10.2f}  {k:>3}  "
                f"{metrics['hit_rate@1']:>7.1%}  "
                f"{metrics['hit_rate@2']:>7.1%}  "
                f"{metrics[f'hit_rate@{k}']:>7.1%}  "
                f"{metrics['mrr']:>7.4f}"
            )
            print(row)
            if best is None or metrics["mrr"] > best["mrr"]:
                best = {**metrics, "k": k, "threshold": thr}

    print("=" * 72)
    best_k = best["k"]
    best_hitrate = best[f"hit_rate@{best_k}"]
    print(
        f"\n✓ Best config by MRR: threshold={best['threshold']:.2f}, k={best_k}"
        f"  →  hit@{best_k}={best_hitrate:.1%}  mrr={best['mrr']:.4f}"
    )


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_report(metrics: dict, details: list[dict]) -> None:
    k = metrics["k"]
    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION REPORT")
    print("=" * 60)
    print(f"  Questions evaluated : {metrics['n_questions']}")
    print(f"  Threshold           : {metrics['threshold']}")
    print(f"  Top-k               : {k}")
    print()
    print(f"  hit-rate@1          : {metrics['hit_rate@1']:.1%}")
    print(f"  hit-rate@2          : {metrics['hit_rate@2']:.1%}")
    print(f"  hit-rate@{k}          : {metrics[f'hit_rate@{k}']:.1%}")
    print(f"  MRR                 : {metrics['mrr']:.4f}")
    print()

    misses = metrics["misses"]
    if misses:
        print(f"  ✗ Missed ({len(misses)} questions):")
        for m in misses:
            expected_short = Path(m["expected"]).stem
            got = [Path(s).stem for s in m["retrieved_sources"]]
            print(f"    [{m['id']:2d}] {m['question'][:55]:<55}")
            print(f"         expected: {expected_short}")
            print(f"         got:      {got or ['(nothing above threshold)']}")
    else:
        print("  ✓ All questions retrieved correctly!")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality.")
    parser.add_argument(
        "--golden",
        default="tests/golden_set.json",
        help="Path to golden set JSON (default: tests/golden_set.json)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=TOP_K,
        help=f"Number of chunks to retrieve (default: {TOP_K})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=RELEVANCE_THRESHOLD,
        help=f"Relevance score threshold (default: {RELEVANCE_THRESHOLD})",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Grid-search threshold and k values, print comparison table",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Save raw results JSON to this path (optional)",
    )
    args = parser.parse_args()

    golden = load_golden_set(args.golden)
    print(f"Loaded {len(golden)} questions from {args.golden}")

    if not os.path.exists(INDEX_DIR):
        print(f"ERROR: FAISS index not found at {INDEX_DIR}. Run the app once first.")
        sys.exit(1)

    if args.tune:
        tune(golden)
        return

    print(f"Loading index … ", end="", flush=True)
    t0 = time.perf_counter()
    vs = build_vectorstore()
    print(f"done ({time.perf_counter()-t0:.1f}s)")

    print(f"Evaluating {len(golden)} questions (threshold={args.threshold}, k={args.k}) …")
    t0 = time.perf_counter()
    metrics, details = evaluate(golden, vs, k=args.k, threshold=args.threshold)
    elapsed = time.perf_counter() - t0

    print_report(metrics, details)
    print(f"\n  Total retrieval time: {elapsed:.1f}s  ({elapsed/len(golden)*1000:.0f} ms/question)")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "metrics": {k: v for k, v in metrics.items() if k != "misses"},
            "details": details,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
