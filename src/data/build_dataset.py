"""
src/data/build_dataset.py

Merge all scraped JSON files, deduplicate by URL hash, and download product images.

Usage:
    python -m src.data.build_dataset
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup

from src.utils.config import (
    DATA_RAW,
    DATA_PROCESSED,
    DATA_IMAGES,
    IMAGE_DOWNLOAD_DELAY,
    IMAGE_TIMEOUT,
    IMAGE_MAX_SIZE,
)
from src.utils.logger import get_logger

log = get_logger("build_dataset")


def load_json_files() -> list[dict[str, Any]]:
    """Load all JSON files from raw data directory."""
    products = []
    json_files = list(DATA_RAW.glob("*.json"))
    
    if not json_files:
        log.warning("No JSON files found in %s", DATA_RAW)
        return products
    
    log.info("Loading %d JSON files from %s", len(json_files), DATA_RAW)
    
    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)
                # Handle both list and dict formats
                items = data if isinstance(data, list) else [data]
                products.extend(items)
                log.debug("Loaded %d products from %s", len(items), json_file.name)
        except (json.JSONDecodeError, IOError) as e:
            log.error("Failed to load %s: %s", json_file.name, e)
    
    log.info("Total products loaded: %d", len(products))
    return products


def deduplicate_by_hash(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Deduplicate products by URL hash.
    Keeps the first occurrence of each unique URL.
    """
    seen_urls = set()
    deduplicated = []
    
    for product in products:
        url = product.get("url", "")
        url_hash = hashlib.sha256(url.encode()).hexdigest()
        
        if url_hash not in seen_urls:
            seen_urls.add(url_hash)
            deduplicated.append(product)
    
    removed = len(products) - len(deduplicated)
    log.info("Deduplication: removed %d duplicates (%d → %d)", 
             removed, len(products), len(deduplicated))
    
    return deduplicated


def download_image(image_url: str, product_id: str) -> str | None:
    """
    Download image from URL and save locally.
    Returns the local filename if successful, None otherwise.
    """
    if not image_url:
        return None
    
    try:
        image_path = DATA_IMAGES / f"{product_id}.jpg"
        
        # Skip if already downloaded
        if image_path.exists():
            return image_path.name
        
        response = httpx.get(
            image_url,
            timeout=IMAGE_TIMEOUT,
            follow_redirects=True,
        )
        response.raise_for_status()
        
        # Check size before saving
        if len(response.content) > IMAGE_MAX_SIZE:
            log.warning("Image %s too large (%d bytes), skipping", 
                       product_id, len(response.content))
            return None
        
        image_path.write_bytes(response.content)
        log.debug("Downloaded image for %s", product_id)
        return image_path.name
        
    except Exception as e:
        log.debug("Failed to download image for %s: %s", product_id, e)
        return None


def build_dataset() -> None:
    """Main pipeline: load, deduplicate, download images, save CSV."""
    log.info("=== Building dataset ===")
    
    # 1. Load and deduplicate
    products = load_json_files()
    if not products:
        log.error("No products loaded. Check scrapers.")
        return
    
    products = deduplicate_by_hash(products)
    
    # 2. Add product IDs and download images
    for i, product in enumerate(products):
        product["product_id"] = f"prod_{i:06d}"
        
        image_url = product.get("image_url")
        if image_url:
            image_filename = download_image(image_url, product["product_id"])
            product["image_filename"] = image_filename or ""
            time.sleep(IMAGE_DOWNLOAD_DELAY)
        else:
            product["image_filename"] = ""
    
    # 3. Save as CSV
    output_csv = DATA_PROCESSED / "products.csv"
    
    if products:
        # Get all unique keys as column headers
        fieldnames = set()
        for product in products:
            fieldnames.update(product.keys())
        fieldnames = sorted(list(fieldnames))
        
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(products)
        
        log.info("Saved %d products to %s", len(products), output_csv)
    else:
        log.error("No products to save")


if __name__ == "__main__":
    build_dataset()
