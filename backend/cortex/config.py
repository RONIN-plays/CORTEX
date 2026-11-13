import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BACKEND_DIR.parent


def _load_env_file(path: Path) -> None:
    """Minimal .env loader: KEY=VALUE lines, real environment variables win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        if key.strip() and value:
            os.environ.setdefault(key.strip(), value)


_load_env_file(PROJECT_DIR / ".env")
_load_env_file(BACKEND_DIR / ".env")

DATA_DIR = Path(os.environ.get("CORTEX_DATA_DIR", BACKEND_DIR / "data"))
DB_PATH = DATA_DIR / "cortex.db"

EMBED_MODEL = os.environ.get("CORTEX_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# Groq-hosted models: one writes answers, one reads images.
LLM_MODEL = os.environ.get("CORTEX_LLM_MODEL", "openai/gpt-oss-120b")
VISION_MODEL = os.environ.get("CORTEX_VISION_MODEL", "qwen/qwen3.8-27b")

# Chunking: sizes are in words.
CHUNK_WORDS = 80
CHUNK_OVERLAP_WORDS = 20

# Retrieval
CANDIDATES_PER_RETRIEVER = 50
RRF_K = 60
CHAT_CONTEXT_CHUNKS = 10

# Graph: each note links to its closest neighbours above a similarity threshold.
# With enough notes, similarity is measured after removing the library-wide average
# (see graph._similarity), where 0 means "no more alike than any two notes".
GRAPH_CENTERING_MIN_NOTES = 8
GRAPH_MIN_SIMILARITY = float(os.environ.get("CORTEX_GRAPH_MIN_SIM", "0.15"))
GRAPH_MIN_SIMILARITY_RAW = 0.65
GRAPH_MAX_LINKS_PER_NOTE = 3

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
