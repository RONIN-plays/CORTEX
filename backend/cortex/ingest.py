import hashlib

from .chunking import Chunk, chunk_text
from .db import Store
from .embeddings import Embedder
from .extract import Extracted, ExtractError, extract_file, extract_url


def ingest(store: Store, embedder: Embedder, extracted: Extracted, source: str | None) -> dict:
    """Chunk, embed and store one extracted document. Identical content is stored once."""
    body = "\n\n".join(text for text, _ in extracted.sections)
    content_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    existing = store.find_by_hash(content_hash)
    if existing is not None:
        return {"id": existing, "title": extracted.title, "duplicate": True}

    chunks: list[Chunk] = []
    for text, page in extracted.sections:
        chunks.extend(chunk_text(text, page=page))
    if not chunks:
        raise ExtractError("Nothing to save: no text was found.")

    # The title and heading give short chunks the context they need to be found.
    passages = [
        "\n".join(filter(None, [extracted.title, chunk.heading, chunk.text])) for chunk in chunks
    ]
    embeddings = embedder.embed_passages(passages)
    doc_id = store.add_document(
        extracted.title, extracted.source_type, source, body, content_hash, chunks, embeddings
    )
    return {"id": doc_id, "title": extracted.title, "chunks": len(chunks), "duplicate": False}


def ingest_text(store: Store, embedder: Embedder, title: str, text: str) -> dict:
    if not text.strip():
        raise ExtractError("The note is empty.")
    extracted = Extracted(title.strip() or "Untitled note", "note", [(text, None)])
    return ingest(store, embedder, extracted, None)


def ingest_file(store: Store, embedder: Embedder, filename: str, data: bytes) -> dict:
    return ingest(store, embedder, extract_file(filename, data), filename)


def ingest_url(store: Store, embedder: Embedder, url: str) -> dict:
    return ingest(store, embedder, extract_url(url), url)
