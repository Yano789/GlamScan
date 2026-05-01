"""
src/retrieval/build_index.py

Build FAISS similarity index from embeddings.
Auto-selects IndexFlatIP (exact) for <10k products or IndexIVFFlat (approximate) for larger.

Usage:
    python -m src.retrieval.build_index
"""

from __future__ import annotations

import numpy as np
import faiss

from src.utils.config import EMBEDDINGS_PATH, FAISS_INDEX_PATH, FAISS_THRESHOLD
from src.utils.logger import get_logger

log = get_logger("build_index")


def build_index() -> None:
    """Load embeddings and build FAISS index."""
    log.info("=== Building FAISS index ===")
    
    # Load embeddings
    if not EMBEDDINGS_PATH.exists():
        log.error("Embeddings file not found: %s", EMBEDDINGS_PATH)
        log.info("Run: python -m src.models.infer_embedder")
        return
    
    embeddings = np.load(EMBEDDINGS_PATH)
    log.info("Loaded embeddings shape: %s", embeddings.shape)
    
    n_products, embedding_dim = embeddings.shape
    
    # Decide index type based on dataset size
    if n_products < FAISS_THRESHOLD:
        log.info("Small dataset (%d products). Using IndexFlatIP (exact).", n_products)
        index = faiss.IndexFlatIP(embedding_dim)
    else:
        log.info("Large dataset (%d products). Using IndexIVFFlat (approximate).", n_products)
        # Use IVF with ~sqrt(n) clusters for good performance
        n_clusters = max(1, int(np.sqrt(n_products)))
        quantizer = faiss.IndexFlatIP(embedding_dim)
        index = faiss.IndexIVFFlat(quantizer, embedding_dim, n_clusters)
    
    # Normalize embeddings for cosine similarity via inner product
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Add to index
    log.info("Adding %d embeddings to index...", n_products)
    index.add(embeddings_norm)
    
    # For IVF, train the index
    if isinstance(index, faiss.IndexIVFFlat):
        log.info("Training IVF index...")
        # Already trained during construction, but ensure it's set up
        index.is_trained = True
    
    # Save index
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    log.info("Saved FAISS index to %s", FAISS_INDEX_PATH)
    
    log.info("=== Index building complete ===")
    log.info("Index type: %s", type(index).__name__)
    log.info("Index size: %d products × %d dimensions", n_products, embedding_dim)


if __name__ == "__main__":
    build_index()
