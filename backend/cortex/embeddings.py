import threading
from typing import Protocol

import numpy as np

from . import config


class Embedder(Protocol):
    def embed_passages(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return (matrix / np.maximum(norms, 1e-12)).astype(np.float32)


class LocalEmbedder:
    """Runs a small ONNX embedding model on this machine. Nothing leaves the computer.

    The model (about 130 MB) is downloaded once on first use and cached.
    """

    def __init__(self, model_name: str = config.EMBED_MODEL):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _get(self):
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding

                self._model = TextEmbedding(self.model_name)
            return self._model

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return _normalize(np.vstack(list(self._get().embed(texts))))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(np.vstack(list(self._get().query_embed([text]))))[0]
