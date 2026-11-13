from typing import Literal

import numpy as np

from . import config
from .db import Store
from .embeddings import Embedder

Mode = Literal["hybrid", "semantic", "keyword"]


def _semantic_ranking(store: Store, embedder: Embedder, query: str, limit: int) -> list[int]:
    chunk_ids, _, vectors = store.embedding_matrix()
    if len(chunk_ids) == 0:
        return []
    scores = vectors @ embedder.embed_query(query)
    top = np.argsort(-scores)[:limit]
    return [int(chunk_ids[i]) for i in top]


def search(
    store: Store,
    embedder: Embedder,
    query: str,
    k: int = 8,
    mode: Mode = "hybrid",
) -> list[dict]:
    """Return the top-k chunks for a query.

    hybrid: reciprocal rank fusion of the semantic and keyword rankings, so a chunk
    scores well if either retriever ranks it highly and best if both do.
    """
    limit = max(k, config.CANDIDATES_PER_RETRIEVER)
    semantic = _semantic_ranking(store, embedder, query, limit) if mode != "keyword" else []
    keyword = store.keyword_search(query, limit) if mode != "semantic" else []

    scores: dict[int, float] = {}
    for ranking in (semantic, keyword):
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (config.RRF_K + rank)

    top = sorted(scores, key=lambda cid: (-scores[cid], cid))[:k]
    rows = store.get_chunks(top)
    semantic_rank = {cid: r for r, cid in enumerate(semantic, start=1)}
    keyword_rank = {cid: r for r, cid in enumerate(keyword, start=1)}
    return [
        {
            **rows[cid],
            "score": round(scores[cid], 5),
            "semantic_rank": semantic_rank.get(cid),
            "keyword_rank": keyword_rank.get(cid),
        }
        for cid in top
        if cid in rows
    ]
