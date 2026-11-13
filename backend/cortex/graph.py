import numpy as np

from . import config
from .db import Store


def _centroids(store: Store) -> tuple[list[int], np.ndarray]:
    """One unit vector per document: the mean of its chunk embeddings."""
    _, doc_ids, vectors = store.embedding_matrix()
    ids = sorted(set(doc_ids.tolist()))
    if not ids:
        return [], np.zeros((0, 0), dtype=np.float32)
    means = np.vstack([vectors[doc_ids == doc_id].mean(axis=0) for doc_id in ids])
    means /= np.maximum(np.linalg.norm(means, axis=1, keepdims=True), 1e-12)
    return ids, means


def _similarity(store: Store) -> tuple[list[int], np.ndarray, float]:
    """Note-to-note similarity matrix and the threshold a link must reach.

    Embeddings of unrelated texts still share a large common component, so raw cosine
    links almost everything. Subtracting the library's mean vector first leaves only what
    makes each note distinctive. That needs a reasonable number of notes to estimate the
    mean, so small libraries fall back to raw cosine with a higher threshold.
    """
    ids, means = _centroids(store)
    if len(ids) < 2:
        return ids, np.zeros((len(ids), len(ids)), dtype=np.float32), 1.0
    threshold = config.GRAPH_MIN_SIMILARITY_RAW
    if len(ids) >= config.GRAPH_CENTERING_MIN_NOTES:
        means = means - means.mean(axis=0)
        means /= np.maximum(np.linalg.norm(means, axis=1, keepdims=True), 1e-12)
        threshold = config.GRAPH_MIN_SIMILARITY
    similarity = means @ means.T
    np.fill_diagonal(similarity, -1.0)
    return ids, similarity, threshold


def build_graph(
    store: Store,
    min_similarity: float | None = None,
    max_links: int = config.GRAPH_MAX_LINKS_PER_NOTE,
) -> dict:
    """Notes are nodes; each note links to its most similar notes above the threshold."""
    documents = {d["id"]: d for d in store.list_documents()}
    ids, similarity, threshold = _similarity(store)
    if min_similarity is not None:
        threshold = min_similarity
    links: dict[tuple[int, int], float] = {}
    for i, doc_id in enumerate(ids):
        for j in np.argsort(-similarity[i])[:max_links]:
            value = float(similarity[i, j])
            if value < threshold:
                break
            links[(min(doc_id, ids[j]), max(doc_id, ids[j]))] = value
    degree: dict[int, int] = {}
    for a, b in links:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
    return {
        "nodes": [
            {
                "id": doc_id,
                "title": doc["title"],
                "source_type": doc["source_type"],
                "chunk_count": doc["chunk_count"],
                "links": degree.get(doc_id, 0),
            }
            for doc_id, doc in documents.items()
        ],
        "links": [
            {"source": a, "target": b, "similarity": round(value, 3)}
            for (a, b), value in sorted(links.items())
        ],
    }


def related_documents(store: Store, doc_id: int, limit: int = 5) -> list[dict]:
    ids, similarity, threshold = _similarity(store)
    if doc_id not in ids:
        return []
    row = similarity[ids.index(doc_id)]
    titles = {d["id"]: d["title"] for d in store.list_documents()}
    return [
        {"id": ids[i], "title": titles[ids[i]], "similarity": round(float(row[i]), 3)}
        for i in np.argsort(-row)[:limit]
        if row[i] >= threshold
    ]
