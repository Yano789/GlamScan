"""
src/data/scrape_amazon.py

Scrapes cosmetic products from Amazon's public search pages.
Uses requests + BeautifulSoup (no API key required).

NOTE: Amazon aggressively blocks scrapers. This implementation:
  - Rotates User-Agent headers
  - Adds human-like delays
  - Uses HTTPX with HTTP/2 support
  - Falls back gracefully when blocked (429 / CAPTCHA detected)

Usage:
    python -m src.data.scrape_amazon --category lipstick --max 100
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

log = get_logger("amazon_scraper")

# ── Header pools ──────────────────────────────────────────────────────────────
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.4; rv:124.0) Gecko/20100101 Firefox/124.0",
]

_ACCEPT_LANGS = ["en-US,en;q=0.9", "en-GB,en;q=0.8", "en-CA,en;q=0.9"]


def _headers() -> dict:
    return {
        "User-Agent":       random.choice(_USER_AGENTS),
        "Accept-Language":  random.choice(_ACCEPT_LANGS),
        "Accept":           "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Encoding":  "gzip, deflate, br",
        "DNT":              "1",
        "Connection":       "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest":   "document",
        "Sec-Fetch-Mode":   "navigate",
        "Sec-Fetch-Site":   "none",
        "Sec-Fetch-User":   "?1",
        "Cache-Control":    "max-age=0",
    }


def _sleep(extra: float = 0.0) -> None:
    delay = random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX) + extra
    time.sleep(delay)


def _is_blocked(html: str) -> bool:
    """Detect CAPTCHA or robot-check pages."""
    markers = [
        "robot check",
        "captcha",
        "enter the characters you see below",
        "sorry, we just need to make sure you're not a robot",
        "api-services-support@amazon.com",
    ]
    lower = html.lower()
    return any(m in lower for m in markers)


# ── URL builder ───────────────────────────────────────────────────────────────
def _search_url(keyword: str, page: int = 1) -> str:
    """
    Amazon beauty search:
    https://www.amazon.com/s?k=lipstick&i=beauty&page=2
    """
    q = keyword.replace(" ", "+")
    return f"https://www.amazon.com/s?k={q}&i=beauty&page={page}"


# ── Parsers ───────────────────────────────────────────────────────────────────
def _parse_search_page(html: str, category: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    products = []

    # Each result card has data-component-type="s-search-result"
    cards = soup.select('[data-component-type="s-search-result"]')
    log.debug("Found %d result cards", len(cards))

    for card in cards:
        try:
            product = _extract_card(card, category)
            if product:
                products.append(product)
        except Exception as exc:
            log.debug("Card parse error: %s", exc)

    return products


def _extract_card(card, category: str) -> dict | None:
    # ASIN
    asin = card.get("data-asin", "")

    # Title
    title_el = (
        card.select_one("h2 a span")
        or card.select_one(".a-size-medium.a-color-base.a-text-normal")
        or card.select_one(".a-size-base-plus.a-color-base.a-text-normal")
    )
    name = title_el.get_text(strip=True) if title_el else ""
    if not name:
        return None

    # Brand  (often in a separate span above the title)
    brand_el = card.select_one(".a-size-base.a-color-secondary")
    brand    = brand_el.get_text(strip=True) if brand_el else ""

    # Price
    price_whole = card.select_one(".a-price-whole")
    price_frac  = card.select_one(".a-price-fraction")
    if price_whole:
        whole = price_whole.get_text(strip=True).replace(",", "").rstrip(".")
        frac  = price_frac.get_text(strip=True) if price_frac else "00"
        price = f"${whole}.{frac}"
    else:
        # Try the "a-offscreen" price (screen-reader price)
        offscreen = card.select_one(".a-price .a-offscreen")
        price = offscreen.get_text(strip=True) if offscreen else ""

    # Rating
    rating_el = card.select_one(".a-icon-star-small .a-icon-alt, .a-icon-star .a-icon-alt")
    rating    = rating_el.get_text(strip=True).split(" ")[0] if rating_el else ""

    # Review count
    reviews_el = card.select_one(".a-size-base[aria-label*='stars']") or \
                 card.select_one('[aria-label*="rating"]')
    reviews    = ""
    if reviews_el:
        match = re.search(r"([\d,]+)", reviews_el.get("aria-label", ""))
        reviews = match.group(1) if match else ""

    # Image
    img_el = card.select_one("img.s-image")
    img    = img_el["src"] if img_el else ""

    # URL
    link_el = card.select_one("h2 a[href], a.a-link-normal[href*='/dp/']")
    href    = link_el["href"] if link_el else f"/dp/{asin}"
    url     = f"https://www.amazon.com{href}" if href.startswith("/") else href
    # Strip tracking query params, keep clean ASIN URL
    url = re.sub(r"\?.*", "", url)

    return {
        "source":       "amazon",
        "category":     category,
        "name":         name,
        "brand":        brand,
        "price":        price,
        "currency":     "USD",
        "rating":       rating,
        "reviews":      reviews,
        "url":          url,
        "image_url":    img,
        "asin":         asin,
    }


# ── Main scrape loop ──────────────────────────────────────────────────────────
def scrape_category(
    category: str,
    max_products: int = 200,
    start_page: int = 1,
) -> Generator[dict, None, None]:
    """Yield product dicts for a given category keyword."""
    fetched          = 0
    page             = start_page
    consecutive_fails = 0

    with httpx.Client(
        timeout=SCRAPE_TIMEOUT,
        follow_redirects=True,
        http2=False,          # Amazon does not always negotiate H2 cleanly
    ) as client:
        while fetched < max_products:
            url = _search_url(category, page)
            log.info("Amazon | page %d | %s", page, url)

            html = None
            for attempt in range(3):
                try:
                    resp = client.get(url, headers=_headers())
                    resp.raise_for_status()
                    html = resp.text
                    break
                except httpx.HTTPStatusError as exc:
                    log.warning("HTTP %s on attempt %d", exc.response.status_code, attempt + 1)
                    _sleep(extra=3.0)
                except httpx.HTTPError as exc:
                    log.warning("Request error: %s", exc)
                    _sleep(extra=2.0)

            if html is None:
                consecutive_fails += 1
                if consecutive_fails >= 3:
                    log.error("Too many consecutive failures — stopping.")
                    break
                continue

            if _is_blocked(html):
                log.warning("Amazon CAPTCHA/block detected on page %d. Backing off 30s.", page)
                time.sleep(30)
                consecutive_fails += 1
                if consecutive_fails >= 2:
                    log.error("Blocked repeatedly — stopping category scrape.")
                    break
                continue

            consecutive_fails = 0
            products = _parse_search_page(html, category)

            if not products:
                log.info("No products on page %d — likely end of results.", page)
                break

            for product in products:
                if fetched >= max_products:
                    break
                yield product
                fetched += 1

            page += 1
            _sleep()

    log.info("Amazon | '%s' | scraped %d products", category, fetched)


def scrape_all(max_per_category: int = 100) -> list[dict]:
    all_products: list[dict] = []

    for cat in PRODUCT_CATEGORIES:
        products = list(scrape_category(cat, max_products=max_per_category))
        all_products.extend(products)

        out_path = DATA_RAW / f"amazon_{cat.replace(' ', '_')}.json"
        out_path.write_text(json.dumps(products, indent=2))
        log.info("Saved %d products → %s", len(products), out_path)

    return all_products


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Amazon beauty products")
    parser.add_argument("--category", default=None)
    parser.add_argument("--max", type=int, default=100)
    args = parser.parse_args()

    if args.category:
        products = list(scrape_category(args.category, max_products=args.max))
        out = DATA_RAW / f"amazon_{args.category.replace(' ', '_')}.json"
        out.write_text(json.dumps(products, indent=2))
        print(f"Saved {len(products)} products to {out}")
    else:
        all_p = scrape_all(max_per_category=args.max)
        print(f"Total: {len(all_p)} products scraped from Amazon")
