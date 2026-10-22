"""
CLI Script: Convert raw .RData files to memory-efficient, subsampled Parquet files.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.convert import process_and_convert_rdata_to_parquet
from src.utils import logger


def main():
    logger.info("Starting .RData -> Parquet conversion & subsetting pipeline...")
    try:
        converted = process_and_convert_rdata_to_parquet()
        logger.info("Successfully converted all dataset files to Parquet:")
        for key, path in converted.items():
            logger.info(f"  - {key}: {path} ({path.stat().st_size / (1024*1024):.2f} MB)")
    except Exception as e:
        logger.error(f"Parquet conversion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
