# 💄 GlamScan

**Image-Based Cosmetic Product Search & Price Comparison**

Upload any cosmetic product photo → GlamScan finds visually similar products across **Sephora** and **Amazon**, sorted by similarity with live prices.

---

## Architecture

```
Image Upload
     │
     ▼
CLIP ViT-B/32  ──────────────────────────────── Text Query
(zero-shot, no fine-tuning)                    (zero-shot)
     │                                               │
     ▼                                               ▼
512-d L2-normalised embedding  ◄──── blend ────  512-d embedding
     │
     ▼
FAISS IVF-Flat Index  (cosine similarity via inner product)
     │
     ▼
Top-K Product Matches
     │
     ▼
Price Comparison Display  (Sephora + Amazon)
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
bash run.sh all
```

This runs all 4 data pipeline steps in order:

| Step | Command | What it does |
|------|---------|--------------|
| 1 | `bash run.sh scrape` | Scrapes Sephora + Amazon (no API key needed) |
| 2 | `bash run.sh build`  | Merges data, downloads product images |
| 3 | `bash run.sh embed`  | Generates CLIP embeddings for all products |
| 4 | `bash run.sh index`  | Builds FAISS similarity index |

### 3. Start the app
```bash
# Terminal 1 — API
bash run.sh api        # → http://localhost:8000
bash run.sh api        # Swagger UI: http://localhost:8000/docs

# Terminal 2 — Web UI
bash run.sh frontend   # → http://localhost:8501
```

---

## Project Structure

```
glamscan/
├── src/
│   ├── data/
│   │   ├── scrape_sephora.py   # Sephora scraper (requests + BS4)
│   │   ├── scrape_amazon.py    # Amazon scraper (requests + BS4)
│   │   └── build_dataset.py    # Merge, clean, download images
│   ├── models/
│   │   ├── embedder.py         # CLIP wrapper (zero-shot)
│   │   └── infer_embedder.py   # Batch embedding generation
│   ├── retrieval/
│   │   ├── build_index.py      # FAISS index builder
│   │   └── search.py           # Similarity search engine
│   ├── api/
│   │   └── app.py              # FastAPI backend
│   └── utils/
│       ├── config.py           # All settings in one place
│       └── logger.py           # Logging
├── frontend/
│   └── app.py                  # Streamlit UI
├── data/
│   ├── raw/                    # Scraped JSON files
│   ├── processed/              # Final CSV dataset
│   └── images/                 # Downloaded product images
├── outputs/
│   ├── embeddings.npy          # CLIP vectors (N × 512)
│   ├── faiss_index.bin         # FAISS index
│   └── metadata.json          # Product metadata
├── requirements.txt
└── run.sh                      # Pipeline runner
```

---

## Search Modes

| Mode | Input | Use case |
|------|-------|----------|
| **Image only** | Product photo | "Find this exact product cheaper" |
| **Text only** | Description string | "matte red lipstick" (zero-shot) |
| **Image + Text** | Photo + description | Most precise — blend both signals |

---

## Scraping Notes

- **No API keys required** — uses public HTML pages only
- **Sephora**: parses JSON embedded in page `<script>` blocks; falls back to HTML card parsing
- **Amazon**: rotates User-Agent headers; detects CAPTCHA and backs off automatically
- Polite random delays (1.5–3.5 s) between requests to avoid bans
- Saves raw JSON per category; rebuild only what you need

---

## Extending

- **Add a new retailer**: implement `scrape_<retailer>.py` following the same generator pattern, add it to `build_dataset.py`
- **Change embedding model**: set `CLIP_MODEL` / `CLIP_PRETRAINED` in `.env` or `config.py` (any `open_clip` model works)
- **Swap to GPU**: embeddings auto-use CUDA if available — no code changes needed
- **Scale up**: for >1M products switch `IndexFlatIP` → `IndexIVFPQ` in `build_index.py`

---

## Planned

- [ ] Shade / colour matching
- [ ] Brand classifier
- [ ] OCR for product label reading
- [ ] Personalised recommendations
- [ ] Mobile app
