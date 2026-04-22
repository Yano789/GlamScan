"""Main entry point for building the cosmetics dataset.

Usage:
    python -m src.data.build_catalog
"""

from __future__ import annotations

import sys

from src.data.build_dataset import build_dataset_from_scrapers
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
	"""Build the cosmetics product catalog."""
	try:
		settings = get_settings()
		
		# Build dataset from scrapers
		catalog_path = build_dataset_from_scrapers(
			output_csv="data/processed/catalog.csv",
			interim_dir="data/interim",
			min_records=500,
			max_records=1500,
			products_per_source=600,
		)
		
		logger.info(f"\n✓ Catalog ready for use: {catalog_path}")
		logger.info("Next steps:")
		logger.info("  1. Use catalog.csv for product recommendations")
		logger.info("  2. Generate embeddings with: python -m src.models.train_embedder")
		logger.info("  3. Build retrieval index for search")
		
	except Exception as e:
		logger.error(f"Error building dataset: {e}", exc_info=True)
		sys.exit(1)


if __name__ == "__main__":
	main()
