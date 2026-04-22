"""Data cleaning and normalization utilities for cosmetic product records."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


# Target schema
PRODUCT_SCHEMA = {
	"product_id": str,
	"name": str,
	"brand": str,
	"category": str,
	"shade": str,
	"price": float,
	"currency": str,
	"image_url": str,
	"product_url": str,
	"retailer": str,
}


def normalize_price(price: Any) -> float | None:
	"""Normalize price to float, handling various formats.
	
	Args:
		price: Price value in various formats
		
	Returns:
		Float price or None if invalid
	"""
	if price is None or pd.isna(price):
		return None
	
	if isinstance(price, (int, float)):
		return float(price) if float(price) > 0 else None
	
	# Try to extract numeric value from string
	if isinstance(price, str):
		# Remove currency symbols and letters
		cleaned = re.sub(r'[^\d.,]', '', price.strip())
		# Replace comma with period for decimal
		cleaned = cleaned.replace(',', '.')
		try:
			value = float(cleaned)
			return value if value > 0 else None
		except (ValueError, AttributeError):
			return None
	
	return None


def normalize_text(text: Any, max_length: int | None = None) -> str | None:
	"""Normalize text field.
	
	Args:
		text: Text to normalize
		max_length: Maximum length (truncate if exceeded)
		
	Returns:
		Cleaned text or None if empty
	"""
	if text is None or pd.isna(text):
		return None
	
	cleaned = str(text).strip()
	if not cleaned:
		return None
	
	if max_length:
		cleaned = cleaned[:max_length]
	
	return cleaned


def normalize_url(url: Any) -> str | None:
	"""Normalize URL field.
	
	Args:
		url: URL to normalize
		
	Returns:
		Valid URL or None
	"""
	if url is None or pd.isna(url):
		return None
	
	url_str = str(url).strip()
	if not url_str or len(url_str) < 10:
		return None
	
	# Ensure URL starts with http
	if not url_str.startswith(('http://', 'https://')):
		url_str = 'https://' + url_str
	
	return url_str


def clean_product_record(record: dict[str, Any]) -> dict[str, Any] | None:
	"""Clean and normalize a single product record.
	
	Args:
		record: Raw product record
		
	Returns:
		Cleaned record or None if invalid
	"""
	if not record:
		return None
	
	cleaned = {}
	
	# Required fields
	product_id = normalize_text(record.get("product_id"))
	if not product_id:
		return None
	cleaned["product_id"] = product_id
	
	name = normalize_text(record.get("name"), max_length=255)
	if not name:
		return None
	cleaned["name"] = name
	
	brand = normalize_text(record.get("brand"), max_length=100)
	if not brand:
		return None
	cleaned["brand"] = brand
	
	category = normalize_text(record.get("category"), max_length=50)
	if not category:
		return None
	cleaned["category"] = category.lower()
	
	shade = normalize_text(record.get("shade"), max_length=100)
	if not shade:
		return None
	cleaned["shade"] = shade
	
	price = normalize_price(record.get("price"))
	if price is None:
		return None
	cleaned["price"] = price
	
	currency = normalize_text(record.get("currency"), max_length=3)
	if not currency:
		currency = "USD"
	cleaned["currency"] = currency.upper()
	
	image_url = normalize_url(record.get("image_url"))
	if not image_url:
		return None
	cleaned["image_url"] = image_url
	
	product_url = normalize_url(record.get("product_url"))
	if not product_url:
		return None
	cleaned["product_url"] = product_url
	
	retailer = normalize_text(record.get("retailer"), max_length=50)
	if not retailer:
		return None
	cleaned["retailer"] = retailer
	
	return cleaned


def clean_records(df: pd.DataFrame) -> pd.DataFrame:
	"""Apply comprehensive cleaning to a dataframe of product records.
	
	Args:
		df: Raw dataframe
		
	Returns:
		Cleaned dataframe with normalized values
	"""
	if df.empty:
		return pd.DataFrame(columns=list(PRODUCT_SCHEMA.keys()))
	
	# Normalize column names
	df.columns = [c.strip().lower() for c in df.columns]
	
	# Clean each record
	cleaned_records = []
	for _, row in df.iterrows():
		record_dict = row.to_dict()
		cleaned = clean_product_record(record_dict)
		if cleaned:
			cleaned_records.append(cleaned)
	
	if not cleaned_records:
		return pd.DataFrame(columns=list(PRODUCT_SCHEMA.keys()))
	
	# Create dataframe from cleaned records
	cleaned_df = pd.DataFrame(cleaned_records)
	
	# Ensure all required columns exist
	for col in PRODUCT_SCHEMA.keys():
		if col not in cleaned_df.columns:
			cleaned_df[col] = None
	
	# Select only required columns in order
	cleaned_df = cleaned_df[list(PRODUCT_SCHEMA.keys())]
	
	# Remove duplicates based on product_id
	cleaned_df = cleaned_df.drop_duplicates(subset=['product_id'], keep='first')
	
	# Sort by product_id for consistency
	cleaned_df = cleaned_df.sort_values('product_id').reset_index(drop=True)
	
	return cleaned_df


def validate_dataset(df: pd.DataFrame) -> dict[str, Any]:
	"""Validate dataset quality and return statistics.
	
	Args:
		df: Dataset to validate
		
	Returns:
		Dictionary with validation statistics
	"""
	stats = {
		"total_records": len(df),
		"missing_values": df.isnull().sum().to_dict(),
		"unique_brands": df["brand"].nunique() if "brand" in df else 0,
		"unique_retailers": df["retailer"].nunique() if "retailer" in df else 0,
		"unique_categories": df["category"].nunique() if "category" in df else 0,
		"price_min": df["price"].min() if "price" in df else None,
		"price_max": df["price"].max() if "price" in df else None,
		"price_mean": df["price"].mean() if "price" in df else None,
	}
	return stats
