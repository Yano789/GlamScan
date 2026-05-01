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
# Note: ViT-B-32 crashes on M2 8GB RAM. Using ViT-B-16 instead.
# Both produce 512-dim embeddings, but with different semantic space.
# Regenerate embeddings on Colab GPU with this model.
CLIP_MODEL = "ViT-B-16"
CLIP_PRETRAINED = "openai"
EMBEDDING_DIM = 512

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

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 1

# ── Frontend ──────────────────────────────────────────────────────────────────
FRONTEND_PORT = 8501
