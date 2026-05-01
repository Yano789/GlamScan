"""
src/data/scrape_sephora.py

Scrapes cosmetic products from Sephora's public website.
Uses httpx with smart HTML parsing and API fallback.

Usage:
    python -m src.data.scrape_sephora --category lipstick --max 100
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Generator

import httpx
from bs4 import BeautifulSoup

from src.utils.config import DATA_RAW, PRODUCT_CATEGORIES, REQUEST_DELAY_MIN, REQUEST_DELAY_MAX, SCRAPE_TIMEOUT
from src.utils.logger import get_logger

log = get_logger("sephora_scraper")

# ── Headers pool (rotate to avoid blocks) ─────────────────────────────────────
_HEADERS_POOL = [
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
    },
    {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
    },
]


def _headers() -> dict:
    return random.choice(_HEADERS_POOL)


def _sleep() -> None:
    time.sleep(random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX))


# ── Sephora search URL builder ────────────────────────────────────────────────
def _search_url(keyword: str, page: int = 1) -> str:
    """
    Sephora search results page:
    https://www.sephora.com/search?keyword=lipstick&currentPage=1
    """
    q = keyword.replace(" ", "+")
    return f"https://www.sephora.com/search?keyword={q}&currentPage={page}"


# ── Parse product grid with Playwright ────────────────────────────────────────
def _parse_products_from_page(html: str, category: str) -> list[dict]:
    """
    Extract products from rendered HTML.
    Sephora uses React with dynamically loaded products.
    """
    soup = BeautifulSoup(html, "html.parser")
    products = []
    
    # Look for product tiles (data attributes vary, try multiple selectors)
    # Modern Sephora uses divs with data-test or role attributes
    product_selectors = [
        '[data-testid*="ProductTile"]',
        '[role="link"][href*="/p/"]',
        'a[href*="/p/"][href*="/product"]',
        '.css-ix8km1',  # Fallback CSS class
    ]
    
    for selector in product_selectors:
        items = soup.select(selector)
        if items:
            log.debug("Found %d products with selector: %s", len(items), selector)
            for item in items:
                product = _extract_product_from_element(item, category)
                if product and product not in products:  # Avoid duplicates
                    products.append(product)
            if products:
                break
    
    return products


def _extract_product_from_element(element, category: str) -> dict | None:
    """Extract product info from a single element."""
    try:
        # Find link to product page
        link_el = element.find("a", href=re.compile(r"/p/\d+"))
        if not link_el:
            link_el = element.find("a")
        
        if not link_el or not link_el.get("href"):
            return None
        
        href = link_el.get("href", "")
        if not href.startswith("/"):
            return None
        
        # Find product name
        name_el = element.find(string=re.compile(r".+")) or element
        name = (name_el.get_text(strip=True) if hasattr(name_el, 'get_text') 
                else str(name_el).strip())[:100]
        
        if not name or len(name) < 2:
            return None
        
        # Find price
        price_el = element.find(string=re.compile(r"\$[\d,]+"))
        price = price_el.strip() if price_el else ""
        
        if not price:
            # Try to find price in any price-related element
            price_candidates = element.find_all(string=re.compile(r"\$"))
            if price_candidates:
                price = price_candidates[0].strip()
        
        # Find image
        img_el = element.find("img")
        image_url = img_el.get("src", "") if img_el else ""
        
        # Extract SKU from URL if available
        sku_match = re.search(r"/p/(\d+)", href)
        sku_id = sku_match.group(1) if sku_match else ""
        
        if not sku_id:
            return None  # Skip products without SKU
        
        return {
            "source": "sephora",
            "category": category,
            "name": name,
            "brand": "",  # Sephora brand info requires deeper parsing
            "price": price,
            "currency": "USD",
            "url": f"https://www.sephora.com{href}",
            "image_url": image_url,
            "sku_id": sku_id,
        }
    except Exception as e:
        log.debug("Error extracting product: %s", e)
        return None


# ── Scrape with httpx (no browser needed) ─────────────────────────────────────
def scrape_category_with_httpx(
    category: str,
    max_products: int = 200,
    start_page: int = 1,
) -> Generator[dict, None, None]:
    """Scrape using httpx with intelligent HTML parsing."""
    fetched = 0
    page_num = start_page
    
    with httpx.Client(timeout=SCRAPE_TIMEOUT, follow_redirects=True) as client:
        while fetched < max_products:
            url = _search_url(category, page_num)
            log.info("Sephora | page %d | %s", page_num, url)
            
            for attempt in range(3):
                try:
                    resp = client.get(url, headers=_headers())
                    resp.raise_for_status()
                    break
                except httpx.HTTPError as exc:
                    log.warning("Attempt %d failed: %s", attempt + 1, exc)
                    _sleep()
                    if attempt == 2:
                        log.error("Giving up on page %d after 3 attempts", page_num)
                        return
            
            products = _parse_products_from_page(resp.text, category)
            
            if not products:
                log.info("No products found on page %d — stopping.", page_num)
                break
            
            for product in products:
                if fetched >= max_products:
                    break
                yield product
                fetched += 1
            
            page_num += 1
            _sleep()
    
    log.info("Sephora | '%s' | scraped %d products", category, fetched)


# ── Main scrape function ──────────────────────────────────────────────────────
def scrape_category(
    category: str,
    max_products: int = 200,
    start_page: int = 1,
) -> Generator[dict, None, None]:
    """Scrape products using httpx."""
    yield from scrape_category_with_httpx(
        category=category,
        max_products=max_products,
        start_page=start_page,
    )


def scrape_all(max_per_category: int = 100) -> list[dict]:
    """Scrape all configured categories."""
    all_products: list[dict] = []

    for cat in PRODUCT_CATEGORIES:
        products = list(scrape_category(cat, max_products=max_per_category))
        all_products.extend(products)

        # Save incrementally
        out_path = DATA_RAW / f"sephora_{cat.replace(' ', '_')}.json"
        out_path.write_text(json.dumps(products, indent=2))
        log.info("Saved %d products → %s", len(products), out_path)

    log.info("Sephora | Total: %d products scraped", len(all_products))
    return all_products


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Sephora products")
    parser.add_argument("--category", default=None, help="Single category to scrape (default: all)")
    parser.add_argument("--max",      type=int, default=100, help="Max products per category")
    args = parser.parse_args()

    if args.category:
        products = list(scrape_category(args.category, max_products=args.max))
        out = DATA_RAW / f"sephora_{args.category.replace(' ', '_')}.json"
        out.write_text(json.dumps(products, indent=2))
        print(f"✓ Saved {len(products)} products to {out}")
    else:
        products = scrape_all(max_per_category=args.max)
        print(f"✓ Scraped {len(products)} total products from Sephora")
