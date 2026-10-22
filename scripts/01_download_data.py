"""
CLI Script: Download raw Tennessee Eastman Process (TEP) dataset from Harvard Dataverse.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.downloader import download_tep_data
from src.utils import logger


def main():
    logger.info("Starting TEP dataset acquisition from Harvard Dataverse (DOI: 10.7910/DVN/6C3JR1)...")
    try:
        downloaded = download_tep_data()
        logger.info("Successfully downloaded all requested dataset files:")
        for key, path in downloaded.items():
            logger.info(f"  - {key}: {path} ({path.stat().st_size / (1024*1024):.2f} MB)")
    except Exception as e:
        logger.error(f"Data acquisition failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
