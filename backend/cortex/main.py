import json
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, graph, ingest, llm
from .db import Store
from .embeddings import Embedder, LocalEmbedder
from .extract import ExtractError
from .search import search


class NoteIn(BaseModel):
    title: str = Field(default="", max_length=300)
    text: str = Field(min_length=1)


class UrlIn(BaseModel):
    url: str = Field(min_length=1, max_length=2000)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatIn(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[ChatMessage] = []


def create_app(store: Store | None = None, embedder: Embedder | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = store or Store(config.DB_PATH)
        app.state.embedder = embedder or LocalEmbedder()
        yield
        if store is None:
            app.state.store.close()

    app = FastAPI(title="Cortex", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _ingest(fn, *args) -> dict:
        try:
            return fn(app.state.store, app.state.embedder, *args)
        except (ExtractError, llm.LLMError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/health")
    def health() -> dict:
        documents, chunks = app.state.store.counts()
        return {
            "ok": True,
            "documents": documents,
            "chunks": chunks,
            "has_api_key": llm.has_credentials(),
            "llm_model": config.LLM_MODEL,
            "embed_model": config.EMBED_MODEL,
        }

    @app.get("/api/documents")
    def list_documents() -> list[dict]:
        return app.state.store.list_documents()

    @app.post("/api/documents/text")
    def add_note(note: NoteIn) -> dict:
        return _ingest(ingest.ingest_text, note.title, note.text)

    @app.post("/api/documents/url")
    def add_url(body: UrlIn) -> dict:
        return _ingest(ingest.ingest_url, body.url.strip())

    @app.post("/api/documents/upload")
    def upload(files: list[UploadFile] = File(...)) -> list[dict]:
        """Ingest each file independently so one bad file does not fail the batch."""
        results = []
        for file in files:
            name = file.filename or "upload"
            data = file.file.read(config.MAX_UPLOAD_BYTES + 1)
            if len(data) > config.MAX_UPLOAD_BYTES:
                results.append({"filename": name, "error": "File is larger than 25 MB."})
                continue
            try:
                result = ingest.ingest_file(app.state.store, app.state.embedder, name, data)
                results.append({"filename": name, **result})
            except (ExtractError, llm.LLMError) as exc:
                results.append({"filename": name, "error": str(exc)})
        return results

    @app.get("/api/documents/{doc_id}")
    def get_document(doc_id: int) -> dict:
        document = app.state.store.get_document(doc_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Note not found.")
        document["related"] = graph.related_documents(app.state.store, doc_id)
        return document

    @app.delete("/api/documents/{doc_id}")
    def delete_document(doc_id: int) -> dict:
        if not app.state.store.delete_document(doc_id):
            raise HTTPException(status_code=404, detail="Note not found.")
        return {"deleted": doc_id}

    @app.get("/api/search")
    def search_notes(
        q: str = Query(min_length=1, max_length=1000),
        mode: Literal["hybrid", "semantic", "keyword"] = "hybrid",
        k: int = Query(default=10, ge=1, le=50),
    ) -> list[dict]:
        return search(app.state.store, app.state.embedder, q, k=k, mode=mode)

    @app.get("/api/graph")
    def get_graph() -> dict:
        return graph.build_graph(app.state.store)

    @app.post("/api/chat")
    async def chat(body: ChatIn) -> StreamingResponse:
        """Server-sent events: one "sources" event, then "delta" events, then "done" or "error"."""
        hits = await run_in_threadpool(
            search,
            app.state.store,
            app.state.embedder,
            body.question,
            config.CHAT_CONTEXT_CHUNKS,
        )
        history = [m.model_dump() for m in body.history]

        async def events():
            yield _sse({"type": "sources", "sources": hits})
            async for event in llm.stream_answer(body.question, hits, history):
                yield _sse(event)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Serve the built frontend, if present, so one process runs the whole app.
    dist = config.PROJECT_DIR / "frontend" / "dist"
    if (dist / "index.html").is_file():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            candidate = (dist / path).resolve()
            if path and candidate.is_file() and candidate.is_relative_to(dist.resolve()):
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")

    return app


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


app = create_app()
