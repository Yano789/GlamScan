"""
src/data/scrape_sephora.py

Scrapes cosmetic products from Sephora's public website.
Uses requests + BeautifulSoup (no API key required).

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
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    },
    {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) "
            "Version/17.4 Safari/605.1.15"
        ),
        "Accept-Language": "en-GB,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    },
]


def _headers() -> dict:
    return random.choice(_HEADERS_POOL)


def _sleep() -> None:
    time.sleep(random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX))


# ── Sephora search URL builder ────────────────────────────────────────────────
def _search_url(keyword: str, page: int = 1) -> str:
    """
    Sephora uses a keyword search page:
    https://www.sephora.com/search?keyword=lipstick&currentPage=1
    """
    q = keyword.replace(" ", "+")
    return f"https://www.sephora.com/search?keyword={q}&currentPage={page}"


# ── Parse a single search results page ───────────────────────────────────────
def _parse_search_page(html: str, category: str) -> list[dict]:
    """
    Extract product cards from a Sephora search results page.
    Sephora renders product data in a <script type="application/json"> block
    labelled 'js-initial-store-state', which contains full product info as JSON.
    Falls back to HTML card parsing if that block is missing.
    """
    soup = BeautifulSoup(html, "html.parser")
    products = []

    # ── Strategy 1: JSON store state (most reliable) ──────────────────────────
    script_tag = soup.find("script", {"id": "linkStore"}) or \
                 soup.find("script", {"type": "application/json"})

    if script_tag and script_tag.string:
        try:
            data = json.loads(script_tag.string)
            # Navigate to product list (path varies by page version)
            items = (
                data.get("page", {})
                    .get("nonLazyLoadedContent", {})
                    .get("searchResults", {})
                    .get("products", [])
            )
            for item in items:
                product = _extract_from_json_item(item, category)
                if product:
                    products.append(product)
            if products:
                return products
        except (json.JSONDecodeError, AttributeError):
            pass

    # ── Strategy 2: HTML card parsing (fallback) ──────────────────────────────
    cards = soup.select('[data-comp="ProductTile"] , .css-ix8km1')
    for card in cards:
        try:
            product = _extract_from_html_card(card, category)
            if product:
                products.append(product)
        except Exception as exc:
            log.debug("Card parse error: %s", exc)

    return products


def _extract_from_json_item(item: dict, category: str) -> dict | None:
    try:
        name  = item.get("displayName", "").strip()
        brand = item.get("brandName", "").strip()
        price = str(item.get("currentSku", {}).get("listPrice", ""))
        url   = "https://www.sephora.com" + item.get("targetUrl", "")
        img   = (item.get("currentSku", {}) or {}).get("skuImages", {}).get("image135", "")
        sku_id = item.get("currentSku", {}).get("skuId", "")

        if not name or not price:
            return None

        return {
            "source":    "sephora",
            "category":  category,
            "name":      name,
            "brand":     brand,
            "price":     _clean_price(price),
            "currency":  "USD",
            "url":       url,
            "image_url": img,
            "sku_id":    str(sku_id),
        }
    except Exception:
        return None


def _extract_from_html_card(card, category: str) -> dict | None:
    name_el  = card.select_one('[data-comp="ProductTile"] a span, .css-0')
    price_el = card.select_one('[data-comp="Price"], .css-slquam')
    brand_el = card.select_one('[data-comp="ProductTile"] span.css-euydo4')
    link_el  = card.select_one("a[href]")
    img_el   = card.select_one("img[src]")

    name  = (name_el.get_text(strip=True)  if name_el  else "")
    price = (price_el.get_text(strip=True) if price_el else "")
    brand = (brand_el.get_text(strip=True) if brand_el else "")
    href  = (link_el["href"]               if link_el  else "")
    img   = (img_el["src"]                 if img_el   else "")

    if not name:
        return None

    return {
        "source":    "sephora",
        "category":  category,
        "name":      name,
        "brand":     brand,
        "price":     _clean_price(price),
        "currency":  "USD",
        "url":       f"https://www.sephora.com{href}" if href.startswith("/") else href,
        "image_url": img,
        "sku_id":    "",
    }


def _clean_price(raw: str) -> str:
    """Return only the first price value (e.g. '$24' from '$24 - $68')."""
    match = re.search(r"\$[\d,]+\.?\d*", raw)
    return match.group(0) if match else raw.strip()


# ── Main scrape loop ──────────────────────────────────────────────────────────
def scrape_category(
    category: str,
    max_products: int = 200,
    start_page: int = 1,
) -> Generator[dict, None, None]:
    """Yield product dicts for a given category keyword."""
    fetched = 0
    page    = start_page

    with httpx.Client(timeout=SCRAPE_TIMEOUT, follow_redirects=True) as client:
        while fetched < max_products:
            url = _search_url(category, page)
            log.info("Sephora | page %d | %s", page, url)

            for attempt in range(3):
                try:
                    resp = client.get(url, headers=_headers())
                    resp.raise_for_status()
                    break
                except httpx.HTTPError as exc:
                    log.warning("Attempt %d failed: %s", attempt + 1, exc)
                    _sleep()
            else:
                log.error("Giving up on page %d", page)
                return

            products = _parse_search_page(resp.text, category)
            if not products:
                log.info("No products found on page %d — stopping.", page)
                break

            for product in products:
                if fetched >= max_products:
                    break
                yield product
                fetched += 1

            page += 1
            _sleep()

    log.info("Sephora | '%s' | scraped %d products", category, fetched)


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
        print(f"Saved {len(products)} products to {out}")
    else:
        all_p = scrape_all(max_per_category=args.max)
        print(f"Total: {len(all_p)} products scraped from Sephora")
