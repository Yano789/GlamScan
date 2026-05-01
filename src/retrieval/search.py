"""
src/retrieval/search.py

Similarity search over the FAISS index.

Supports:
  - Image query  (PIL Image or bytes)
  - Text query   (zero-shot, e.g. "matte red lipstick")
  - Combined     (text + image, averaged embeddings)
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import faiss
import numpy as np
from PIL import Image

from src.models.embedder import get_embedder
from src.utils.config import FAISS_INDEX_PATH, METADATA_PATH, TOP_K
from src.utils.logger import get_logger

log = get_logger("search")


@dataclass
class SearchResult:
    rank:        int
    score:       float          # cosine similarity (0–1)
    product_id:  str
    name:        str
    brand:       str
    category:    str
    price:       str
    price_usd:   float | None
    source:      str
    url:         str
    image_url:   str
    rating:      str
    reviews:     str


class GlamSearchEngine:
    """
    Loads the FAISS index and metadata once, then serves queries.
    Thread-safe for reading (FAISS read-only searches are thread-safe).
    """

    def __init__(self) -> None:
        self._index    : faiss.Index | None = None
        self._metadata : list[dict]          = []
        self._n_products = 0
        self._embedding_dim = 512

    def _load(self) -> None:
        if self._index is not None:
            return  # already loaded

        if not FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {FAISS_INDEX_PATH}. "
                "Run `python -m src.retrieval.build_index` first."
            )
        if not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Metadata not found at {METADATA_PATH}. "
                "Run `python -m src.models.infer_embedder` first."
            )

        self._index    = faiss.read_index(str(FAISS_INDEX_PATH))
        self._metadata = json.loads(METADATA_PATH.read_text())
        self._n_products = len(self._metadata)
        self._embedding_dim = self._index.d if hasattr(self._index, 'd') else 512
        log.info("Search engine ready: %d products", self._n_products)

    @property
    def n_products(self) -> int:
        """Number of products in the index."""
        if self._index is None:
            self._load()
        return self._n_products
    
    @property
    def embedding_dim(self) -> int:
        """Embedding dimension."""
        if self._index is None:
            self._load()
        return self._embedding_dim

    # ── Query helpers ──────────────────────────────────────────────────────────
    def _query(self, vec: np.ndarray, top_k: int) -> list[SearchResult]:
        self._load()
        vec = vec.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)

        scores, indices = self._index.search(vec, top_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self._metadata):
                continue
            m = self._metadata[idx]
            results.append(SearchResult(
                rank       = rank,
                score      = float(score),
                product_id = m.get("product_id", ""),
                name       = m.get("name", ""),
                brand      = m.get("brand", ""),
                category   = m.get("category", ""),
                price      = m.get("price", ""),
                price_usd  = m.get("price_usd"),
                source     = m.get("source", ""),
                url        = m.get("url", ""),
                image_url  = m.get("image_url", ""),
                rating     = m.get("rating", ""),
                reviews    = m.get("reviews", ""),
            ))
        return results

    # ── Public API ─────────────────────────────────────────────────────────────
    def search_by_image(
        self,
        image: Union[Image.Image, bytes],
        top_k: int = TOP_K,
    ) -> list[SearchResult]:
        t0 = time.time()
        embedder = get_embedder()
        log.info(f"Embedder loaded: {time.time() - t0:.2f}s")
        
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")
        
        t1 = time.time()
        vec = embedder.embed_image(image)
        log.info(f"Image embedding: {time.time() - t1:.2f}s")
        
        t2 = time.time()
        results = self._query(vec, top_k)
        log.info(f"FAISS query: {time.time() - t2:.2f}s")
        
        return results

    def search_by_text(
        self,
        query: str,
        top_k: int = TOP_K,
    ) -> list[SearchResult]:
        embedder = get_embedder()
        vec = embedder.embed_text(query)
        return self._query(vec, top_k)

    def search_by_combined(
        self,
        image: Union[Image.Image, bytes],
        text:  str,
        top_k: int = TOP_K,
        text_weight: float = 0.5,
    ) -> list[SearchResult]:
        """Blend image and text embeddings for richer retrieval."""
        embedder = get_embedder()
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")
        img_vec  = embedder.embed_image(image)
        text_vec = embedder.embed_text(text)
        # text_weight + (1 - text_weight) = 1
        vec      = (1 - text_weight) * img_vec + text_weight * text_vec
        return self._query(vec, top_k)


# ── Module-level singleton ─────────────────────────────────────────────────────
_engine: GlamSearchEngine | None = None


def get_engine() -> GlamSearchEngine:
    global _engine
    if _engine is None:
        _engine = GlamSearchEngine()
    return _engine
