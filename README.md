# Cortex

A local-first "second brain". Save notes, PDFs, web pages and images, then search them by meaning and ask questions that are answered from your own material, with links back to the source.

- **Ask**: answers grounded in your notes, each claim cited to the passage it came from.
- **Search**: hybrid search that combines meaning (embeddings) with keywords (BM25).
- **Library**: upload files, write notes, or save a web page by URL.
- **Graph**: notes are linked automatically when they are about similar things.

Your notes, the search index and the embedding model all stay on your machine. Only the passages retrieved for a question (and any image you upload) are sent to the LLM provider.

## How it works

```
file / note / URL
      │  extract text (Markdown, PDF, HTML; images are transcribed by a vision model)
      ▼
   chunks of ~80 words, overlapping
      │  embed locally (bge-small-en-v1.5, ONNX)
      ▼
   SQLite: documents, chunks + vectors, FTS5 keyword index
      │
question ──► semantic top 50 ─┐
         └─► keyword  top 50 ─┴─► reciprocal rank fusion ──► top 10 passages
                                                                  │
                                              LLM writes an answer citing [1], [2] …
```

| Part | Choice |
|---|---|
| Backend | Python, FastAPI |
| Storage | SQLite (vectors as blobs, FTS5 for keywords) |
| Embeddings | `BAAI/bge-small-en-v1.5` via fastembed, runs on CPU |
| Answers | Groq API (`openai/gpt-oss-120b`), streamed |
| Image reading | Groq API (`qwen/qwen3.8-27b`) |
| Frontend | React, TypeScript, Vite |

## Setup

Requires Python 3.12+ and Node 20+.

```powershell
# 1. Backend
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-dev.txt

# 2. Frontend
cd ..\frontend
npm install
npm run build

# 3. API key (free): create one at https://console.groq.com/keys
#    then put it in a file named .env in the project root:
#    GROQ_API_KEY=your-key
```

Without a key, search, the library and the graph still work; only answers and image uploads need it.

## Run

```powershell
cd backend
.\.venv\Scripts\python -m uvicorn cortex.main:app --port 8000
```

Open http://127.0.0.1:8000. The first start downloads the embedding model (about 130 MB) once.

To try it with sample content, load 26 short notes on software, learning and a few everyday topics:

```powershell
.\.venv\Scripts\python -m eval.load_samples
```

For frontend development, run `npm run dev` in `frontend/` (port 5173, proxies `/api` to the backend).

## Evaluation

`backend/eval/dataset.json` holds 26 sample notes and 52 questions, each labelled with the note that answers it. Half the questions reuse the note's wording and half are paraphrased to share few keywords with it.

```powershell
cd backend
.\.venv\Scripts\python -m eval.run_eval
```

| Mode | Recall@1 | Recall@3 | Recall@5 | MRR@10 |
|---|---|---|---|---|
| Keyword only (BM25) | 76.9% | 92.3% | 96.2% | 0.857 |
| Semantic only | 78.8% | 92.3% | 96.2% | 0.862 |
| **Hybrid** | **86.5%** | **98.1%** | **98.1%** | **0.926** |

Recall@k is the share of questions whose answer note appears in the top k results. Hybrid search cuts first-result misses by about 40% relative to either method alone (12 or 11 misses down to 7 of 52).

Limits of these numbers: the dataset is small and synthetic, and the chunk size was chosen using the same questions, so expect somewhat lower figures on your own notes. The evaluation measures retrieval only, not the quality of the written answers.

## Tests

```powershell
cd backend
.\.venv\Scripts\python -m pytest
```

Tests run offline with a stand-in embedder and never call the LLM.

## Deploy to Vercel

`vercel.json` deploys the repo as one Vercel project with two services:

| Service | Folder | Public path |
|---|---|---|
| `frontend` | `frontend/` (Vite static build) | everything except `/api` |
| `backend` | `backend/` (FastAPI, entrypoint `cortex.main:app`) | `/api/*` |

The browser calls `/api/...` on the same domain, so the services need no bindings. The backend receives the full path, including `/api`.

1. Import the GitHub repo into Vercel. It reads `vercel.json` from the repo root.
2. In the project settings, add the environment variable `GROQ_API_KEY`.
3. Deploy. To test locally first, run `vercel dev` in the project root.

Limits on Vercel:

- **Notes don't persist.** Functions can only write to `/tmp`, which is wiped whenever an instance is recycled and isn't shared between instances. The page shows a warning about this. Lasting storage needs a hosted database.
- **Uploads are capped at 4 MB per file** (Vercel rejects request bodies over 4.5 MB).
- **The first request after a cold start is slow**, because the embedding model (about 64 MB) downloads into `/tmp`.

## Settings

Optional lines for `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | none | Enables answers and image reading |
| `CORTEX_LLM_MODEL` | `openai/gpt-oss-120b` | Groq model that writes answers |
| `CORTEX_VISION_MODEL` | `qwen/qwen3.8-27b` | Groq model that reads images |
| `CORTEX_DATA_DIR` | `backend/data` | Where the database lives |
| `CORTEX_GRAPH_MIN_SIM` | `0.15` | How similar two notes must be to be linked |

## Known limits

- Search compares the query against every stored vector. That is fast up to tens of thousands of chunks; beyond that it needs an approximate index.
- Scanned PDFs with no text layer are rejected; upload the pages as images instead.
- The embedding model is English-focused.
- There are no user accounts. Run it on your own machine only.
