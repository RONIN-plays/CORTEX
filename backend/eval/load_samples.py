"""Load the sample notes into your real Cortex library, to try the app before adding your own.

    python -m eval.load_samples
"""

import json
from pathlib import Path

from cortex import config
from cortex.db import Store
from cortex.embeddings import LocalEmbedder
from cortex.ingest import ingest_text


def main() -> None:
    dataset = json.loads((Path(__file__).resolve().parent / "dataset.json").read_text("utf-8"))
    store, embedder = Store(config.DB_PATH), LocalEmbedder()
    added = 0
    for doc in dataset["documents"]:
        added += not ingest_text(store, embedder, doc["title"], doc["text"])["duplicate"]
    print(f"Added {added} sample notes ({len(dataset['documents']) - added} were already there).")


if __name__ == "__main__":
    main()
