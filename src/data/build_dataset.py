"""Build final cosmetics dataset from multiple sources.

This module combines data from various scrapers/sources, cleans it,
and produces a unified, high-quality product catalog.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.data.clean_data import clean_records, validate_dataset
from src.data.scrape_douglas import scrape_amazon
from src.data.scrape_sephora import scrape_sephora
from src.data.scrape_ulta import scrape_ulta
from src.utils.helpers import ensure_parent_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_dataset_from_scrapers(
	output_csv: str,
	interim_dir: str = "data/interim",
	min_records: int = 500,
	max_records: int = 1500,
	products_per_source: int = 600,
) -> Path:
	"""Build dataset by scraping multiple retailers and combining data.
	
	Args:
		output_csv: Path to save final dataset
		interim_dir: Directory to save intermediate files
		min_records: Minimum records required
		max_records: Maximum records to include
		products_per_source: Products to fetch from each source
		
	Returns:
		Path to output CSV
	"""
	logger.info("=" * 60)
	logger.info("Building Cosmetics Dataset")
	logger.info("=" * 60)
	
	ensure_parent_dir(output_csv)
	interim_path = Path(interim_dir)
	interim_path.mkdir(parents=True, exist_ok=True)
	
	all_records = []
	
	# Scrape from multiple sources
	logger.info(f"\nFetching data from Sephora ({products_per_source} products)...")
	sephora_data = scrape_sephora(limit=products_per_source)
	all_records.extend(sephora_data)
	logger.info(f"✓ Sephora: {len(sephora_data)} products")
	
	logger.info(f"\nFetching data from Ulta ({products_per_source} products)...")
	ulta_data = scrape_ulta(limit=products_per_source)
	all_records.extend(ulta_data)
	logger.info(f"✓ Ulta: {len(ulta_data)} products")
	
	logger.info(f"\nFetching data from Amazon ({products_per_source} products)...")
	amazon_data = scrape_amazon(limit=products_per_source)
	all_records.extend(amazon_data)
	logger.info(f"✓ Amazon: {len(amazon_data)} products")
	
	logger.info(f"\nTotal raw records: {len(all_records)}")
	
	# Convert to dataframe
	if not all_records:
		logger.error("No data collected from any source!")
		sys.exit(1)
	
	df = pd.DataFrame(all_records)
	
	# Save raw intermediate file
	raw_interim_path = interim_path / "raw_combined.csv"
	df.to_csv(raw_interim_path, index=False)
	logger.info(f"Saved raw combined data: {raw_interim_path}")
	
	# Clean data
	logger.info("\nCleaning and normalizing data...")
	cleaned_df = clean_records(df)
	logger.info(f"After cleaning: {len(cleaned_df)} valid records")
	
	if len(cleaned_df) < min_records:
		logger.warning(f"Dataset has {len(cleaned_df)} records, below minimum {min_records}")
	
	# Save cleaned intermediate file
	cleaned_interim_path = interim_path / "cleaned_combined.csv"
	cleaned_df.to_csv(cleaned_interim_path, index=False)
	logger.info(f"Saved cleaned data: {cleaned_interim_path}")
	
	# Limit to max_records
	if len(cleaned_df) > max_records:
		logger.info(f"Limiting dataset to {max_records} records (was {len(cleaned_df)})")
		cleaned_df = cleaned_df.sample(n=max_records, random_state=42).reset_index(drop=True)
	
	# Validate final dataset
	logger.info("\nValidating final dataset...")
	stats = validate_dataset(cleaned_df)
	logger.info(f"✓ Total records: {stats['total_records']}")
	logger.info(f"✓ Unique brands: {stats['unique_brands']}")
	logger.info(f"✓ Unique retailers: {stats['unique_retailers']}")
	logger.info(f"✓ Price range: ${stats['price_min']:.2f} - ${stats['price_max']:.2f}")
	logger.info(f"✓ Average price: ${stats['price_mean']:.2f}")
	
	# Save final dataset
	output_path = Path(output_csv)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	cleaned_df.to_csv(output_path, index=False)
	logger.info(f"\n✓ Final catalog saved: {output_path}")
	logger.info(f"✓ Size: {len(cleaned_df)} products")
	
	logger.info("\n" + "=" * 60)
	logger.info("Dataset Build Complete!")
	logger.info("=" * 60)
	
	return output_path


def build_dataset(interim_dir: str, output_csv: str) -> Path:
	"""Combine interim CSV files into a single processed dataset.
	
	Legacy function for combining pre-existing intermediate files.
	
	Args:
		interim_dir: Directory containing intermediate CSV files
		output_csv: Path to save final dataset
		
	Returns:
		Path to output CSV
	"""
	files = sorted(Path(interim_dir).glob("*.csv"))
	
	if not files:
		logger.warning(f"No CSV files found in {interim_dir}")
		return Path(output_csv)
	
	logger.info(f"Combining {len(files)} intermediate files...")
	frames = [pd.read_csv(path) for path in files]
	dataset = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
	
	# Clean combined data
	dataset = clean_records(dataset)
	
	out_path = Path(output_csv)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	dataset.to_csv(out_path, index=False)
	
	logger.info(f"✓ Dataset saved: {out_path} ({len(dataset)} records)")
	return out_path
