"""
src/api/app.py

FastAPI backend for GlamScan with /search endpoints.

Run:
    uvicorn src.api.app:app --reload
    # Or: bash run.sh api
"""

from __future__ import annotations

import os
import io
import time
import gc
from typing import Optional

# Set PyTorch memory allocation to be more conservative
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Fix OpenMP conflicts on macOS

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from src.retrieval.search import GlamSearchEngine, SearchResult
from src.utils.logger import get_logger

log = get_logger("api")

app = FastAPI(
    title="GlamScan",
    description="Image-based cosmetic product search & price comparison",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize search engine (singleton)
_engine: Optional[GlamSearchEngine] = None
_embedder_cache = None  # Keep strong reference to embedder


def get_engine() -> GlamSearchEngine:
    """Lazy-load search engine."""
    global _engine
    if _engine is None:
        log.info("Initializing GlamSearchEngine...")
        gc.collect()  # Clean memory before loading
        _engine = GlamSearchEngine()
    return _engine


# Pre-load embedder at startup
@app.on_event("startup")
async def startup_embedder():
    """Load embedder at startup to avoid lazy loading during requests."""
    global _embedder_cache
    try:
        log.info("Pre-loading CLIP embedder at startup...")
        gc.collect()  # Clean up memory before loading
        
        # Import here to apply env vars first
        import torch
        torch.set_num_threads(1)  # Limit PyTorch threads
        
        from src.models.embedder import CLIPEmbedder
        _embedder_cache = CLIPEmbedder()
        log.info("✅ Embedder loaded successfully at startup")
    except Exception as e:
        log.error("❌ Failed to load embedder at startup: %s", e)
        # Don't crash API, just log - we can still do text search
        log.warning("API will work but image search may fail")



# ── Request/Response Models ────────────────────────────────────────────────────


class SearchByTextRequest(BaseModel):
    query: str
    k: int = 10


class SearchByBothRequest(BaseModel):
    query: str
    k: int = 10
    text_weight: float = 0.5  # 0.5 = equal text/image weight


class SearchResultResponse(BaseModel):
    rank: int
    score: float
    product_id: str
    name: str
    brand: str
    category: str
    price: str
    price_usd: Optional[float]
    source: str
    url: str
    image_url: str
    rating: str
    reviews: str


# ── Health Check ───────────────────────────────────────────────────────────────


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        engine = get_engine()
        return {
            "status": "ok",
            "n_products": engine.n_products,
            "embedding_dim": engine.embedding_dim,
        }
    except Exception as e:
        log.error("Health check failed: %s", e)
        raise HTTPException(status_code=503, detail=str(e))


# ── Search by Image ────────────────────────────────────────────────────────────


@app.post("/search/image")
async def search_by_image(
    file: UploadFile = File(...),
    k: int = 10,
) -> list[SearchResultResponse]:
    """
    Search by uploading a product image.
    
    Args:
        file: Image file (JPEG/PNG)
        k: Number of results to return
    
    Returns:
        List of similar products with similarity scores
    """
    try:
        if k <= 0 or k > 100:
            raise ValueError("k must be between 1 and 100")
        
        t0 = time.time()
        # Read image
        content = await file.read()
        log.info(f"File read: {time.time() - t0:.2f}s")
        
        t1 = time.time()
        img = Image.open(io.BytesIO(content)).convert("RGB")
        log.info(f"Image decode: {time.time() - t1:.2f}s")
        
        t2 = time.time()
        engine = get_engine()
        log.info(f"Engine load: {time.time() - t2:.2f}s")
        
        t3 = time.time()
        log.info("Starting embedder...")
        results = engine.search_by_image(img, top_k=k)
        log.info(f"Embedder + search: {time.time() - t3:.2f}s")
        
        return [SearchResultResponse(**result.__dict__) for result in results]
        
    except Exception as e:
        log.error("Search by image failed: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


# ── Search by Text ────────────────────────────────────────────────────────────


@app.post("/search/text")
async def search_by_text(
    request: SearchByTextRequest,
) -> list[SearchResultResponse]:
    """
    Search by text query (zero-shot).
    
    Args:
        query: Text description (e.g., "matte red lipstick")
        k: Number of results to return
    
    Returns:
        List of similar products with similarity scores
    """
    try:
        if not request.query or not request.query.strip():
            raise ValueError("Query cannot be empty")
        
        if request.k <= 0 or request.k > 100:
            raise ValueError("k must be between 1 and 100")
        
        t0 = time.time()
        engine = get_engine()
        log.info(f"Engine load: {time.time() - t0:.2f}s")
        
        t1 = time.time()
        results = engine.search_by_text(request.query, top_k=request.k)
        log.info(f"Text search: {time.time() - t1:.2f}s")
        
        return [SearchResultResponse(**result.__dict__) for result in results]
        
    except Exception as e:
        log.error("Search by text failed: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


# ── Search by Combined (Image + Text) ────────────────────────────────────────


@app.post("/search/combined")
async def search_by_both(
    file: UploadFile = File(...),
    query: str = "",
    k: int = 10,
    text_weight: float = 0.5,
) -> list[SearchResultResponse]:
    """
    Search by both image and text query (blended).
    
    Args:
        file: Image file
        query: Text description
        k: Number of results to return
        text_weight: Weight for text embedding (0-1, default 0.5)
    
    Returns:
        List of similar products with blended similarity scores
    """
    try:
        if k <= 0 or k > 100:
            raise ValueError("k must be between 1 and 100")
        
        if not (0 <= text_weight <= 1):
            raise ValueError("text_weight must be between 0 and 1")
        
        # Read image
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
        
        engine = get_engine()
        results = engine.search_by_combined(
            img,
            query,
            top_k=k,
            text_weight=text_weight,
        )
        
        return [SearchResultResponse(**result.__dict__) for result in results]
        
    except Exception as e:
        log.error("Search by combined failed: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
