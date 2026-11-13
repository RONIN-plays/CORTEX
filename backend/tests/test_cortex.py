"""Tests run offline: a deterministic bag-of-words embedder stands in for the real model."""

import hashlib
import json
import re

import numpy as np
import pytest
from fastapi.testclient import TestClient

from cortex import llm
from cortex.chunking import chunk_text
from cortex.db import Store, fts_query
from cortex.extract import ExtractError, extract_file, html_to_text
from cortex.graph import build_graph
from cortex.ingest import ingest_text
from cortex.main import create_app
from cortex.search import search


class FakeEmbedder:
    DIM = 256

    def _one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.DIM, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM] += 1.0
        return vector / max(np.linalg.norm(vector), 1e-12)

    def embed_passages(self, texts):
        return np.vstack([self._one(t) for t in texts])

    def embed_query(self, text):
        return self._one(text)


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def library(store, embedder):
    ids = {
        "bread": ingest_text(store, embedder, "Bread", "Sourdough needs flour, water, salt and a starter.")["id"],
        "raft": ingest_text(store, embedder, "Raft", "A leader is elected by a majority of votes in each term.")["id"],
        "loaf": ingest_text(store, embedder, "Baking", "Sourdough loaf: flour, water, salt and starter, then bake.")["id"],
    }
    return store, ids


# ---- chunking ---------------------------------------------------------------


def test_short_text_is_one_chunk():
    chunks = chunk_text("Just one paragraph.")
    assert [c.text for c in chunks] == ["Just one paragraph."]


def test_headings_start_new_chunks_and_are_recorded():
    chunks = chunk_text("# One\n\nalpha\n\n## Two\n\nbeta")
    assert [(c.heading, c.text) for c in chunks] == [("One", "alpha"), ("Two", "beta")]


def test_long_text_is_split_within_the_size_limit_and_loses_nothing():
    paragraphs = [" ".join(f"p{p}w{w}" for w in range(40)) + "." for p in range(12)]
    chunks = chunk_text("\n\n".join(paragraphs), max_words=100, overlap_words=40)
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 140 for c in chunks)
    joined = " ".join(c.text for c in chunks)
    assert all(p in joined for p in paragraphs)
    # consecutive chunks share the overlap paragraph
    assert chunks[0].text.split("\n\n")[-1] == chunks[1].text.split("\n\n")[0]


def test_a_single_huge_paragraph_is_still_split():
    chunks = chunk_text(" ".join(["word"] * 1000), max_words=100)
    assert len(chunks) == 10


def test_page_number_is_kept():
    assert chunk_text("text", page=7)[0].page == 7


# ---- storage and search -----------------------------------------------------


def test_fts_query_survives_punctuation_and_operators():
    assert fts_query('what is "NEAR" AND (cpp*)?') == '"near" OR "cpp"'
    assert fts_query("?!") is None


def test_duplicate_content_is_stored_once(store, embedder):
    first = ingest_text(store, embedder, "A", "same text")
    second = ingest_text(store, embedder, "B", "same text")
    assert second == {"id": first["id"], "title": "B", "duplicate": True}
    assert store.counts() == (1, 1)


@pytest.mark.parametrize("mode", ["keyword", "semantic", "hybrid"])
def test_search_finds_the_matching_note(library, embedder, mode):
    store, ids = library
    hits = search(store, embedder, "who is elected leader?", k=3, mode=mode)
    assert hits[0]["doc_id"] == ids["raft"]
    assert hits[0]["title"] == "Raft"


def test_hybrid_reports_both_ranks(library, embedder):
    store, _ = library
    top = search(store, embedder, "sourdough flour water", k=1, mode="hybrid")[0]
    assert top["semantic_rank"] is not None and top["keyword_rank"] is not None


def test_search_on_empty_library(store, embedder):
    assert search(store, embedder, "anything", mode="hybrid") == []


def test_delete_removes_note_from_every_index(library, embedder):
    store, ids = library
    assert store.delete_document(ids["raft"])
    for mode in ("keyword", "semantic", "hybrid"):
        assert all(h["doc_id"] != ids["raft"] for h in search(store, embedder, "leader votes", mode=mode))
    assert not store.delete_document(ids["raft"])


def test_graph_links_similar_notes_only(library):
    store, ids = library
    graph = build_graph(store, min_similarity=0.3)
    assert len(graph["nodes"]) == 3
    assert [(l["source"], l["target"]) for l in graph["links"]] == [(ids["bread"], ids["loaf"])]


# ---- extraction -------------------------------------------------------------


def test_markdown_title_comes_from_first_heading():
    extracted = extract_file("notes.md", b"# My Title\n\nbody")
    assert (extracted.title, extracted.source_type) == ("My Title", "markdown")


def test_html_drops_scripts_and_keeps_headings():
    title, text = html_to_text(
        "<html><head><title>T</title><script>x()</script></head>"
        "<body><nav>menu</nav><h1>Head</h1><p>Hello <b>world</b></p></body></html>"
    )
    assert title == "T"
    assert text == "# Head\n\nHello world"


def test_binary_file_is_rejected_with_a_clear_message():
    with pytest.raises(ExtractError, match="Unsupported file type"):
        extract_file("thing.bin", b"\xff\xfe\x00\x81")


def test_image_without_api_key_explains_why(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(llm.LLMError, match="API key"):
        extract_file("shot.png", b"\x89PNG")


# ---- prompt construction ----------------------------------------------------


def test_messages_number_the_sources_and_end_with_the_question():
    hits = [{"title": "Raft", "heading": "Election", "page": None, "text": "majority wins"}]
    history = [{"role": "assistant", "content": "orphan"}, {"role": "user", "content": "hi"},
               {"role": "assistant", "content": "hello"}]
    messages = llm.build_messages("who wins?", hits, history)
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert '<source id="1" title="Raft" section="Election">' in messages[-1]["content"]
    assert messages[-1]["content"].endswith("Question: who wins?")


# ---- HTTP API ---------------------------------------------------------------


@pytest.fixture
def client(store, embedder, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with TestClient(create_app(store, embedder)) as c:
        yield c


def test_api_round_trip(client):
    note = client.post("/api/documents/text", json={"title": "Raft", "text": "A leader is elected by majority."}).json()
    upload = client.post(
        "/api/documents/upload",
        files=[("files", ("bread.md", b"# Bread\n\nflour and water", "text/markdown")),
               ("files", ("bad.bin", b"\xff\xfe\x00\x81", "application/octet-stream"))],
    ).json()
    assert upload[0]["title"] == "Bread" and "error" in upload[1]

    assert client.get("/api/health").json()["documents"] == 2
    assert {d["title"] for d in client.get("/api/documents").json()} == {"Raft", "Bread"}

    hits = client.get("/api/search", params={"q": "elected leader"}).json()
    assert hits[0]["doc_id"] == note["id"]

    document = client.get(f"/api/documents/{note['id']}").json()
    assert document["chunks"][0]["text"] == "A leader is elected by majority."

    assert client.delete(f"/api/documents/{note['id']}").status_code == 200
    assert client.get(f"/api/documents/{note['id']}").status_code == 404


def test_empty_note_is_rejected(client):
    assert client.post("/api/documents/text", json={"title": "x", "text": "   "}).status_code == 422


def test_chat_without_key_streams_sources_then_a_clear_error(client):
    client.post("/api/documents/text", json={"title": "Raft", "text": "A leader is elected by majority."})
    response = client.post("/api/chat", json={"question": "who is leader?"})
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    assert [e["type"] for e in events] == ["sources", "error"]
    assert events[0]["sources"][0]["title"] == "Raft"
    assert "GROQ_API_KEY" in events[1]["message"]
