"""Ulta Beauty data collection module.

In production, this would use Ulta's API or web scraping.
For demo purposes, generates realistic synthetic data.
"""

from __future__ import annotations

from typing import Any

from src.data.dataset_generator import generate_lipstick_product


def scrape_ulta(limit: int = 500) -> list[dict[str, Any]]:
	"""Fetch Ulta Beauty lipstick products.
	
	Args:
		limit: Maximum number of products to return
		
	Returns:
		List of product dictionaries with Ulta data
	"""
	# In production, this would make real API calls or use web scraping
	# For demo: generate realistic data
	products = []
	for _ in range(limit):
		product = generate_lipstick_product(retailer="ulta")
		products.append(product)
	return products


if __name__ == "__main__":
	rows = scrape_ulta()
	print(f"Fetched {len(rows)} Ulta products")
