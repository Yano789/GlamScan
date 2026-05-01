# 💄 GlamScan

**Image-Based Cosmetic Product Search Engine**

Upload any cosmetic product photo → GlamScan detects the product type and finds visually similar products from Amazon, sorted by color shade and similarity.

---

## Features

✨ **Smart Product Detection** — Automatically detects if you uploaded lipstick, blush, foundation, etc. using CLIP zero-shot classification

🎨 **Color-Aware Matching** — Extracts dominant color from your image and prioritizes matches with similar shades

🔍 **Category-Filtered Search** — Only shows products from the same category (no eyeshadows when searching for blush!)

⚡ **GPU Accelerated** — Uses NVIDIA GPUs with mixed precision (fp16) for 2-3x faster inference

📊 **1515+ Product Index** — Curated Amazon dataset across 8 product categories

---

## Architecture

```
Image Upload
     │
     ▼
CLIP ViT-L-14 + GPU Acceleration  ──────────────── Text Query
(768-d embeddings, fp16 mixed precision)        (zero-shot)
     │                                               │
     ├─► Product Type Detection (zero-shot)        │
     │   (lipstick? blush? foundation?)            │
     │                                               │
     ▼                                               ▼
768-d L2-normalized embedding  ◄──── blend ────  768-d embedding
     │
     ▼
FAISS IndexFlatIP (cosine similarity via inner product)
     │
     ▼
Top-K Results → Category Filter → Color Matching → Ranked Results
     │
     ▼
Product Matches Display  (from Amazon)
```

---

## Quick Start

### Requirements

- **Python 3.10+**
- **NVIDIA GPU** (RTX 4060 or better; fallback to CPU but slower)
- **8GB+ VRAM** (for ViT-L-14 model with mixed precision)

### 1. Install dependencies

```bash
git clone <your-repo-url>
cd GlamScan
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
bash run.sh all
```

This runs all 4 data pipeline steps in order:

| Step | Command | What it does | Time |
|------|---------|--------------|------|
| 1 | `bash run.sh scrape` | Scrapes Amazon (500/category) | ~10-15m |
| 2 | `bash run.sh build`  | Merges data, downloads images, deduplicates | ~5m |
| 3 | `bash run.sh embed`  | Generates ViT-L-14 embeddings (GPU) | ~10-15m |
| 4 | `bash run.sh index`  | Builds FAISS index | ~1m |

**Total pipeline time: ~30-40 minutes**

### 3. Start the app

```bash
# Terminal 1 — API
bash run.sh api        # → http://localhost:8000
                       # Swagger UI: http://localhost:8000/docs

# Terminal 2 — Web UI  
bash run.sh frontend   # → http://localhost:8501
```

---

## Project Structure

```
GlamScan/
├── src/
│   ├── data/
│   │   ├── scrape_amazon.py    # Amazon scraper (500/category)
│   │   └── build_dataset.py    # Merge, deduplicate, download images
│   ├── models/
│   │   ├── embedder.py         # CLIP wrapper (ViT-L-14, GPU, fp16)
│   │   └── infer_embedder.py   # Batch embedding generation
│   ├── retrieval/
│   │   ├── build_index.py      # FAISS index builder
│   │   └── search.py           # Similarity search + filtering
│   ├── api/
│   │   └── app.py              # FastAPI backend
│   └── utils/
│       ├── config.py           # Centralized config (model, search params)
│       └── logger.py           # Logging
├── frontend/
│   └── app.py                  # Streamlit UI
├── data/
│   ├── raw/                    # Scraped JSON files
│   ├── processed/              # Final CSV dataset (products.csv)
│   └── images/                 # Downloaded product images (~1500)
├── outputs/
│   ├── embeddings.npy          # CLIP vectors (1515 × 768)
│   ├── faiss_index.bin         # FAISS index
│   └── metadata.json           # Product metadata
├── requirements.txt
├── run.sh                      # Pipeline orchestrator
└── README.md
```

---

## Search Modes

| Mode | Input | Best for |
|------|-------|----------|
| **Image only** | Product photo | "Show me similar lipsticks to this shade" |
| **Text only** | Description | "matte red lipstick under $20" |
| **Image + Text** | Photo + description | Most accurate — combines visual + text signals |

**Smart Features:**
- **Auto-detection**: Image search automatically detects product type
- **Color extraction**: Analyzes image to find dominant color and matching shades
- **Category filtering**: Heavily penalizes products from wrong categories
- **Shade matching**: Prioritizes products with similar colors

---

## Configuration

All settings in `src/utils/config.py`:

```python
# ── CLIP Model ────
CLIP_MODEL = "ViT-L-14"              # 768-d embeddings
CLIP_PRETRAINED = "openai"           # Pre-trained weights
USE_MIXED_PRECISION = True            # fp16 on CUDA (faster, less VRAM)
EMBEDDING_BATCH_SIZE = 64             # Batch size for GPU

# ── Search ────
TOP_K = 10                            # Return top-10 results
MIN_SIMILARITY = 0.70                 # Filter below 70% similarity
CATEGORY_PENALTY = 0.2                # 80% score reduction for wrong category
```

---

## Data & Deduplication

**Dataset:**
- **Source**: Amazon (no Sephora due to JS rendering limitations in WSL)
- **Scraping**: 500 products/category × 8 categories = 4000 raw products
- **Deduplication**: URL-based hashing removes 60% duplicates → **1515 unique products**
- **Categories**: lipstick, mascara, foundation, eyeshadow, blush, concealer, primer, setting_spray

**Deduplication logic**: Products with identical URLs are removed; keeps first occurrence

---

## Scraping Notes

- **No API keys required** — uses public Amazon pages only
- **Rate limiting**: Random 0.5–2s delays between requests to be respectful
- **Image downloads**: Cached locally; skips if already downloaded
- **Image size limit**: 10MB max per image to avoid storage bloat
- **Error resilience**: Gracefully handles 404s, timeouts, malformed data

---

## GPU & Performance

**GPU Support:**
- ✅ Automatic CUDA detection — uses GPU if available
- ✅ Mixed precision (fp16) — ~2-3x faster inference, ~50% less VRAM
- ✅ Batch processing — embeddings generated in batches of 64

**Tested Setup:**
- GPU: NVIDIA RTX 4060 (8GB VRAM)
- Framework: PyTorch + OpenCLIP
- Model: ViT-L-14 (768-d)
- Batch inference: 64 products/batch → ~2 sec
- Full pipeline: 1515 products → ~15 min embedding time

**CPU Fallback:**
- Works on CPU but significantly slower (~2-3x)
- Not recommended for large datasets

---

## API Endpoints

### `POST /search/image`
Upload a product image → Get similar matches
```bash
curl -X POST http://localhost:8000/search/image \
  -F "image=@lipstick.jpg" \
  -F "top_k=10"
```

### `POST /search/text`
Text query → Get matching products
```bash
curl -X POST http://localhost:8000/search/text \
  -H "Content-Type: application/json" \
  -d '{"query": "red matte lipstick", "top_k": 10}'
```

### `POST /search/combined`
Image + Text → Blended search
```bash
curl -X POST http://localhost:8000/search/combined \
  -F "image=@lipstick.jpg" \
  -F "query=matte red" \
  -F "top_k=10"
```

---

## Troubleshooting

**"CUDA out of memory"**
- Reduce `EMBEDDING_BATCH_SIZE` in `config.py` (try 32 or 16)
- Check no other GPU processes running (`nvidia-smi`)

**"No results returned"**
- Increase `MIN_SIMILARITY` threshold (start at 0.65)
- Check if product is in supported categories

**"Wrong product type detected"**
- Product type detection uses CLIP zero-shot (90%+ accurate but not perfect)
- Try text search with explicit category instead: "red lipstick" instead of image-only

**Scraper not finding products**
- Amazon changes page structure frequently — may need HTML updates
- Check logs: `tail -f logs/scrape.log`

---

## Future Enhancements

- [ ] Add more retailers (Ulta, Nordstrom)
- [ ] Fine-tune CLIP on beauty product dataset
- [ ] Color swatches & shade matching UI
- [ ] Price history tracking
- [ ] User favorites & wishlist
- [ ] Advanced filters (brand, price range, skin tone compatibility)

---

## License

MIT

---
