# Cosmetics Dataset Documentation

## Overview

The cosmetics dataset (`data/processed/catalog.csv`) contains **1,500 lipstick products** from three major retailers: Sephora, Ulta Beauty, and Amazon. This dataset is specifically designed for product recommendation systems, embeddings, and retrieval-augmented generation (RAG) use cases.

## Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Products** | 1,500 |
| **Unique Brands** | 29 |
| **Retailers** | 3 (Sephora, Ulta Beauty, Amazon) |
| **Price Range** | $5.01 - $149.80 |
| **Average Price** | $48.73 |
| **Complete Records** | 100% |

## Schema

Each product record contains the following fields:

```json
{
  "product_id": "unique_identifier_string",
  "name": "Full product name including brand and finish",
  "brand": "Brand name (e.g., MAC, Urban Decay)",
  "category": "lipstick",
  "shade": "Color shade (e.g., Ruby Red, Ballet Pink)",
  "price": 29.50,
  "currency": "USD",
  "image_url": "https://...",
  "product_url": "https://...",
  "retailer": "Retailer name (Sephora, Ulta Beauty, or Amazon Beauty)"
}
```

## Data Quality

✅ **100% Complete Records** - All required fields are populated  
✅ **No Duplicates** - Each product_id is unique  
✅ **No Missing Values** - Every record is fully normalized  
✅ **Balanced Distribution** - Even split across retailers (~500 each)  

## Retailer Distribution

- **Sephora**: 497 products (33%)
- **Ulta Beauty**: 492 products (33%)
- **Amazon Beauty**: 511 products (34%)

## Top Brands

The dataset includes 29 premium cosmetics brands:

1. Guerlain (62 products)
2. Huda Beauty (61 products)
3. NARS (61 products)
4. Lancome (60 products)
5. Too Faced (59 products)
... and 24 more

## Shade Categories & Distribution

### Color Categories
- **Red Shades**: Ruby Red, Crimson, Cherry Red, Classic Red, etc.
- **Pink Shades**: Ballet Pink, Blush Pink, Coral Pink, Fuchsia, etc.
- **Nude Shades**: Beige, Caramel, Champagne, Tan, etc.
- **Burgundy/Wine**: Deep reds and wine tones
- **Berry Shades**: Berry, Blueberry, Blackberry, Raspberry, etc.
- **Coral Shades**: Coral, Peach Coral, Orange Coral, Salmon Coral
- **Orange Shades**: Tangerine, Pumpkin, Apricot, Sunset Orange
- **Brown Shades**: Espresso, Chocolate, Caramel Brown, Tan Brown, Mocha
- **Plum/Purple**: Plum, Purple Plum, Aubergine, Deep Plum

### Top 10 Most Common Shades
1. Plum (65 products)
2. Deep Plum (48 products)
3. Peach Coral (48 products)
4. Aubergine (47 products)
5. Pumpkin (47 products)
6. Purple Plum (45 products)
7. Tangerine (43 products)
8. Tan Brown (43 products)
9. Wine (42 products)
10. Berry (42 products)

## Usage Examples

### Load the Dataset

```python
import pandas as pd

catalog = pd.read_csv('data/processed/catalog.csv')
print(f"Loaded {len(catalog)} products")
```

### Filter by Retailer

```python
sephora_products = catalog[catalog['retailer'] == 'Sephora']
print(f"Sephora has {len(sephora_products)} lipsticks")
```

### Filter by Price Range

```python
budget_friendly = catalog[(catalog['price'] >= 10) & (catalog['price'] <= 30)]
luxury = catalog[catalog['price'] > 70]
```

### Filter by Brand

```python
mac_products = catalog[catalog['brand'] == 'MAC']
urban_decay = catalog[catalog['brand'] == 'Urban Decay']
```

### Find Products by Shade

```python
red_lipsticks = catalog[catalog['shade'].str.contains('Red', case=False)]
pink_shades = catalog[catalog['shade'].str.contains('Pink', case=False)]
```

### Sample Products

```python
# Random sample
sample = catalog.sample(n=10, random_state=42)

# By brand
nars_sample = catalog[catalog['brand'] == 'NARS'].head(5)
```

## Data Cleaning & Normalization

The dataset went through comprehensive cleaning:

1. **Text Normalization**: Whitespace trimmed, consistent capitalization
2. **Price Validation**: All prices are positive floats
3. **URL Validation**: All URLs start with http/https and are well-formed
4. **Deduplication**: Duplicate product_ids removed
5. **Schema Enforcement**: All required fields present and typed correctly

### Cleaning Process

```
Raw Data (1,800 records)
    ↓
Normalization & Validation
    ↓
Deduplicated Data (1,800 records valid)
    ↓
Random Sampling (1,500 final records)
    ↓
Cleaned Catalog (data/processed/catalog.csv)
```

## Integration Points

### Product Embeddings

The catalog is designed to work with CLIP embeddings for:
- Product image similarity
- Text-based search
- Recommendation systems

```bash
# Generate embeddings
python -m src.models.train_embedder
```

This will create:
- `outputs/embeddings.npy` - Image embeddings for all products
- `outputs/faiss_index.bin` - FAISS index for fast similarity search
- `outputs/embeddings_metadata.json` - Mapping between embeddings and products

### Retrieval Index

Use the catalog with the retrieval system:

```python
from src.retrieval.search import ProductSearch

search = ProductSearch('data/processed/catalog.csv', 'outputs/faiss_index.bin')
results = search.find_similar(product_id='sephora_9b4036c00a27', top_k=5)
```

### API Integration

The catalog can be served via the REST API:

```bash
# Start API server
python -m src.api.app
```

Then query products:
```bash
curl http://localhost:8000/api/products/search?query=red%20lipstick
```

## File Organization

```
data/
├── processed/
│   └── catalog.csv              # Final clean dataset (1,500 products)
├── interim/
│   ├── raw_combined.csv         # Raw combined data before cleaning
│   └── cleaned_combined.csv     # Cleaned but not deduplicated
├── raw/
│   └── (source data goes here)
└── images/
    └── (product images for embedding generation)
```

## Quality Metrics

- **Completeness**: 100% (no missing values)
- **Uniqueness**: 100% (no duplicate product_ids)
- **Validity**: 100% (all URLs and prices validated)
- **Consistency**: 100% (uniform schema across all records)

## Future Enhancements

Potential improvements to the dataset:

1. **Add product descriptions** for better embeddings
2. **Include ratings/reviews** for popularity metrics
3. **Add product formulation data** (long-wear, waterproof, etc.)
4. **Expand to other product categories** (foundations, eyeshadow, etc.)
5. **Add temporal data** (seasonal availability, launch dates)

## Notes

- This dataset is synthetic/semi-realistic for demonstration purposes
- Product URLs and image URLs are templated (not all links may be active)
- Real-world datasets would include additional metadata and validation
- The data generation process ensures diversity across brands, prices, and shades

## Support

For issues or questions:

1. Check `data/interim/` directory for intermediate processing files
2. Review `src/data/clean_data.py` for cleaning logic
3. See `src/data/dataset_generator.py` for data generation details

---

**Generated**: April 22, 2026  
**Dataset Version**: 1.0  
**Total Records**: 1,500 lipstick products
