"""Generate realistic cosmetics product data for demo purposes.

This module creates synthetic data simulating real product catalogs from
Sephora, Ulta, and other retailers for testing and demonstration.
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta
from typing import Any

# Realistic cosmetics data
BRANDS = [
    "MAC", "Urban Decay", "Too Faced", "Charlotte Tilbury", "NARS", 
    "Fenty Beauty", "Huda Beauty", "KKW Beauty", "Kylie Cosmetics",
    "Anastasia Beverly Hills", "Stila", "Benefit", "Clinique", "Estée Lauder",
    "Lancome", "Dior", "Giorgio Armani", "YSL Beauty", "Guerlain",
    "Chanel", "Sisley", "Bobbi Brown", "Trish McEvoy", "Laura Mercier",
    "Smashbox", "Bare Minerals", "IT Cosmetics", "Tarte", "Caudalie"
]

LIPSTICK_SHADES = {
    "Red": ["Ruby Red", "Crimson", "Cherry Red", "Classic Red", "Brick Red", 
            "Dark Red", "Blood Red", "Fire Red", "Hot Red", "True Red"],
    "Pink": ["Ballet Pink", "Blush Pink", "Coral Pink", "Dusty Pink", "Fuchsia",
             "Hot Pink", "Mauve Pink", "Nude Pink", "Pale Pink", "Rose"],
    "Nude": ["Beige", "Caramel", "Champagne", "Nude", "Tan", "Warm Nude", 
             "Cool Nude", "Peachy Nude", "Rosy Nude", "Sand"],
    "Burgundy": ["Burgundy", "Wine", "Maroon", "Oxblood", "Plum"],
    "Berry": ["Berry", "Blueberry", "Blackberry", "Raspberry", "Mulberry"],
    "Coral": ["Coral", "Peach Coral", "Orange Coral", "Salmon Coral"],
    "Orange": ["Tangerine", "Pumpkin", "Apricot", "Sunset Orange"],
    "Brown": ["Espresso", "Chocolate", "Caramel Brown", "Tan Brown", "Mocha"],
    "Plum": ["Plum", "Purple Plum", "Deep Plum", "Aubergine"],
}

FINISHES = ["Matte", "Satin", "Glossy", "Metallic", "Shimmer", "Cream", "Liquid Matte"]

RETAILERS = {
    "sephora": {
        "name": "Sephora",
        "url_base": "https://www.sephora.com/product/",
        "currency": "USD"
    },
    "ulta": {
        "name": "Ulta Beauty",
        "url_base": "https://www.ulta.com/product/",
        "currency": "USD"
    },
    "amazon": {
        "name": "Amazon Beauty",
        "url_base": "https://www.amazon.com/s?k=",
        "currency": "USD"
    }
}

PRICE_RANGES = {
    "budget": (5, 15),
    "mid": (15, 35),
    "premium": (35, 70),
    "luxury": (70, 150),
}


def generate_lipstick_product(
    brand: str | None = None,
    retailer: str | None = None,
    product_id: str | None = None,
) -> dict[str, Any]:
    """Generate a single realistic lipstick product record.
    
    Args:
        brand: Brand name (random if None)
        retailer: Retailer key - 'sephora', 'ulta', or 'amazon' (random if None)
        product_id: Custom product ID (generated if None)
        
    Returns:
        Dictionary with product data matching the schema
    """
    if brand is None:
        brand = random.choice(BRANDS)
    
    if retailer is None:
        retailer = random.choice(list(RETAILERS.keys()))
    
    if product_id is None:
        product_id = f"{retailer}_{uuid.uuid4().hex[:12]}"
    
    # Generate product details
    shade_category = random.choice(list(LIPSTICK_SHADES.keys()))
    shade = random.choice(LIPSTICK_SHADES[shade_category])
    finish = random.choice(FINISHES)
    
    product_name = f"{brand} {finish} Lipstick in {shade}"
    
    # Price based on tier
    price_tier = random.choice(list(PRICE_RANGES.keys()))
    min_price, max_price = PRICE_RANGES[price_tier]
    price = round(random.uniform(min_price, max_price), 2)
    
    # Build URLs
    retailer_info = RETAILERS[retailer]
    image_id = uuid.uuid4().hex[:16]
    product_url = f"{retailer_info['url_base']}{brand.replace(' ', '-')}-{shade.replace(' ', '-')}"
    image_url = f"https://images.{retailer}.com/product-images/{image_id}.jpg"
    
    return {
        "product_id": product_id,
        "name": product_name,
        "brand": brand,
        "category": "lipstick",
        "shade": shade,
        "finish": finish,
        "price": price,
        "currency": retailer_info["currency"],
        "image_url": image_url,
        "product_url": product_url,
        "retailer": retailer_info["name"],
        "retailer_code": retailer,
        "in_stock": random.choice([True, True, True, False]),  # 75% in stock
        "rating": round(random.uniform(3.5, 5.0), 1) if random.random() > 0.2 else None,
        "review_count": random.randint(0, 500) if random.random() > 0.3 else 0,
        "long_wear": random.choice([True, False]),
        "waterproof": random.choice([True, False]),
        "shade_category": shade_category,
    }


def generate_lipstick_catalog(
    num_products: int = 1000,
    retailers: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Generate a realistic lipstick catalog.
    
    Args:
        num_products: Number of products to generate
        retailers: List of retailer codes to use (uses all if None)
        
    Returns:
        List of product dictionaries
    """
    if retailers is None:
        retailers = list(RETAILERS.keys())
    
    products = []
    used_ids = set()
    
    for _ in range(num_products):
        # Occasionally repeat brands/shades across retailers (realistic)
        if random.random() < 0.3 and products:
            prev_product = random.choice(products)
            brand = prev_product["brand"]
        else:
            brand = None
        
        retailer = random.choice(retailers)
        
        # Ensure unique product IDs
        while True:
            product = generate_lipstick_product(brand=brand, retailer=retailer)
            if product["product_id"] not in used_ids:
                used_ids.add(product["product_id"])
                break
        
        products.append(product)
    
    return products
