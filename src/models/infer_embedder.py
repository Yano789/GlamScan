"""
src/models/infer_embedder.py

Batch embedding generation for all downloaded product images.
Outputs embeddings.npy and metadata.json to the outputs directory.

Usage:
    python -m src.models.infer_embedder
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.models.embedder import CLIPEmbedder, get_embedder
from src.utils.config import DATA_PROCESSED, DATA_IMAGES, EMBEDDINGS_PATH, METADATA_PATH
from src.utils.logger import get_logger

log = get_logger("infer_embedder")


def load_products_csv() -> list[dict]:
    """Load products from CSV file."""
    import csv
    
    csv_file = DATA_PROCESSED / "products.csv"
    if not csv_file.exists():
        log.error("Products CSV not found: %s", csv_file)
        return []
    
    products = []
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        products = list(reader)
    
    log.info("Loaded %d products from CSV", len(products))
    return products


def infer_embeddings() -> None:
    """Generate embeddings for all products with images."""
    log.info("=== Generating embeddings ===")
    
    # Load products
    products = load_products_csv()
    if not products:
        log.error("No products to embed")
        return
    
    # Initialize embedder
    embedder = get_embedder()
    
    embeddings = []
    metadata = []
    failed_count = 0
    
    log.info("Processing %d products", len(products))
    
    for i, product in enumerate(products):
        product_id = product.get("product_id")
        image_filename = product.get("image_filename")
        
        if not image_filename:
            log.debug("Product %s has no image, skipping", product_id)
            failed_count += 1
            continue
        
        image_path = DATA_IMAGES / image_filename
        
        if not image_path.exists():
            log.warning("Image not found: %s", image_path)
            failed_count += 1
            continue
        
        try:
            # Load and embed image
            img = Image.open(image_path).convert("RGB")
            embedding = embedder.embed_image(img)
            
            embeddings.append(embedding)
            metadata.append({
                "product_id": product_id,
                "name": product.get("name", ""),
                "brand": product.get("brand", ""),
                "category": product.get("category", ""),
                "price": product.get("price", ""),
                "price_usd": float(product.get("price_usd", 0)) if product.get("price_usd") else None,
                "source": product.get("source", ""),
                "url": product.get("url", ""),
                "image_url": product.get("image_url", ""),
                "rating": product.get("rating", ""),
                "reviews": product.get("reviews", ""),
            })
            
            if (i + 1) % 100 == 0:
                log.info("Processed %d / %d products", i + 1, len(products))
            
        except Exception as e:
            log.error("Failed to embed %s: %s", product_id, e)
            failed_count += 1
    
    if not embeddings:
        log.error("No embeddings generated")
        return
    
    # Save embeddings and metadata
    embeddings_array = np.array(embeddings, dtype=np.float32)
    log.info("Embeddings shape: %s", embeddings_array.shape)
    
    np.save(EMBEDDINGS_PATH, embeddings_array)
    log.info("Saved embeddings to %s", EMBEDDINGS_PATH)
    
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("Saved metadata to %s", METADATA_PATH)
    
    log.info("=== Embedding complete ===")
    log.info("Successfully embedded: %d", len(embeddings))
    log.info("Failed: %d", failed_count)


if __name__ == "__main__":
    infer_embeddings()
