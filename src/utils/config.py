"""
src/utils/config.py

Centralized configuration for the GlamScan pipeline.
All settings in one place for easy tuning.
"""

from __future__ import annotations

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_IMAGES = DATA_DIR / "images"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Ensure directories exist
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
DATA_IMAGES.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ── CLIP Model ─────────────────────────────────────────────────────────────────
# ViT-L-14 is optimal for 8GB GPUs (RTX 4060, RTX 3060).
# 768-dim embeddings with excellent accuracy.
# For 10GB+ GPUs, use ViT-bigG-14 (1280-dim, highest accuracy).
CLIP_MODEL = "ViT-L-14"
CLIP_PRETRAINED = "openai"
EMBEDDING_DIM = 768

# ── GPU Optimization ──────────────────────────────────────────────────────────
# Increase batch size for GPU to maximize throughput
EMBEDDING_BATCH_SIZE = 64  # Increase if GPU has >8GB VRAM, decrease for <8GB
# Use mixed precision (fp16) for faster inference on supported GPUs
USE_MIXED_PRECISION = True  # torch.float16 vs torch.float32

# ── FAISS Index ────────────────────────────────────────────────────────────────
FAISS_INDEX_PATH = OUTPUTS_DIR / "faiss_index.bin"
METADATA_PATH = OUTPUTS_DIR / "metadata.json"
EMBEDDINGS_PATH = OUTPUTS_DIR / "embeddings.npy"

# ── Scraping ───────────────────────────────────────────────────────────────────
REQUEST_DELAY = 1.0  # seconds between requests
REQUEST_DELAY_MIN = 0.5
REQUEST_DELAY_MAX = 3.0
SCRAPE_TIMEOUT = 30  # seconds
AMAZON_BLOCK_BACKOFF = 30  # seconds to wait if blocked

# ── Product Categories ────────────────────────────────────────────────────────
PRODUCT_CATEGORIES = [
    "lipstick",
    "mascara",
    "foundation",
    "eyeshadow",
    "blush",
    "concealer",
    "primer",
    "setting_spray",
]

# ── Image Download ────────────────────────────────────────────────────────────
IMAGE_DOWNLOAD_DELAY = 0.5  # seconds between downloads
IMAGE_TIMEOUT = 10  # seconds per download
IMAGE_MAX_SIZE = 10 * 1024 * 1024  # 10 MB

# ── Search ────────────────────────────────────────────────────────────────────
TOP_K = 10  # Return top-K similar products
FAISS_THRESHOLD = 10000  # Use approximate index (IVFFlat) if products > threshold
MIN_SIMILARITY = 0.70  # Filter out results below 70% similarity for image search (relaxed with more data)

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 1

# ── Frontend ──────────────────────────────────────────────────────────────────
FRONTEND_PORT = 8501
