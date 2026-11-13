"""Measure retrieval quality on the sample dataset.

    python -m eval.run_eval

Indexes the sample notes into a throwaway in-memory database, runs every test
question through keyword, semantic and hybrid search, and reports:

  Recall@k  share of questions whose answer note appears in the top k results
  MRR       mean of 1/rank of the answer note within the top 10 (0 if absent)
"""

import json
import sys
import time
from pathlib import Path

from cortex.db import Store
from cortex.embeddings import LocalEmbedder
from cortex.ingest import ingest_text
from cortex.search import search

EVAL_DIR = Path(__file__).resolve().parent
MODES = ("keyword", "semantic", "hybrid")
DEPTH = 10


def answer_rank(hits: list[dict], doc_id: int) -> int | None:
    """Rank (1-based) of the first result that comes from the expected note."""
    for rank, hit in enumerate(hits, start=1):
        if hit["doc_id"] == doc_id:
            return rank
    return None


def main() -> int:
    dataset = json.loads((EVAL_DIR / "dataset.json").read_text(encoding="utf-8"))
    store, embedder = Store(":memory:"), LocalEmbedder()

    started = time.perf_counter()
    doc_ids = {
        doc["id"]: ingest_text(store, embedder, doc["title"], doc["text"])["id"]
        for doc in dataset["documents"]
    }
    index_seconds = time.perf_counter() - started
    documents, chunks = store.counts()

    questions = dataset["questions"]
    ranks: dict[str, list[int | None]] = {mode: [] for mode in MODES}
    latency: dict[str, float] = {}
    for mode in MODES:
        started = time.perf_counter()
        for item in questions:
            hits = search(store, embedder, item["q"], k=DEPTH, mode=mode)
            ranks[mode].append(answer_rank(hits, doc_ids[item["doc"]]))
        latency[mode] = (time.perf_counter() - started) / len(questions) * 1000

    def metrics(mode: str) -> dict:
        found = ranks[mode]
        n = len(found)
        return {
            "recall_at_1": sum(r is not None and r <= 1 for r in found) / n,
            "recall_at_3": sum(r is not None and r <= 3 for r in found) / n,
            "recall_at_5": sum(r is not None and r <= 5 for r in found) / n,
            "mrr_at_10": sum(1 / r for r in found if r) / n,
            "avg_query_ms": round(latency[mode], 1),
        }

    results = {
        "documents": documents,
        "chunks": chunks,
        "questions": len(questions),
        "embedding_model": embedder.model_name,
        "index_seconds": round(index_seconds, 2),
        "modes": {mode: metrics(mode) for mode in MODES},
        "misses_at_1": {
            mode: [
                {"question": item["q"], "expected": item["doc"], "rank": rank}
                for item, rank in zip(questions, ranks[mode])
                if rank != 1
            ]
            for mode in MODES
        },
    }
    (EVAL_DIR / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"{documents} notes, {chunks} chunks, {len(questions)} questions "
          f"({embedder.model_name}, indexed in {index_seconds:.1f}s)\n")
    print(f"{'mode':<10}{'Recall@1':>10}{'Recall@3':>10}{'Recall@5':>10}{'MRR@10':>10}{'ms/query':>10}")
    for mode in MODES:
        m = results["modes"][mode]
        print(f"{mode:<10}{m['recall_at_1']:>10.1%}{m['recall_at_3']:>10.1%}"
              f"{m['recall_at_5']:>10.1%}{m['mrr_at_10']:>10.3f}{m['avg_query_ms']:>10.1f}")
    print(f"\nFull results, including every missed question: {EVAL_DIR / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
