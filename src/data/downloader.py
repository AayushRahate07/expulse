"""
Data Downloader for Tennessee Eastman Process (TEP) dataset files from Harvard Dataverse.
"""

from pathlib import Path
from typing import Dict, Any, Optional

import requests
from tqdm import tqdm

from src.config import DATAVERSE_BASE_URL, RAW_FILES, DATA_RAW_DIR
from src.utils import logger, verify_md5, timeit


def download_file(
    url: str,
    output_path: Path,
    expected_md5: Optional[str] = None,
    expected_size: Optional[int] = None
) -> Path:
    """
    Downloads a single file from URL with streaming, progress bar, and MD5 validation.
    Skips if file exists and MD5 matches.
    """
    if output_path.exists():
        if expected_md5 and verify_md5(output_path, expected_md5):
            logger.info(f"File '{output_path.name}' already exists and passed MD5 check. Skipping download.")
            return output_path
        elif not expected_md5:
            logger.info(f"File '{output_path.name}' already exists. Skipping download.")
            return output_path
        else:
            logger.warning(f"File '{output_path.name}' exists but failed MD5 verification. Re-downloading.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading from '{url}' to '{output_path}'...")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers, stream=True, timeout=60)
    response.raise_for_status()

    total_bytes = int(response.headers.get("content-length", expected_size or 0))

    with open(output_path, "wb") as f, tqdm(
        total=total_bytes,
        unit="B",
        unit_scale=True,
        desc=output_path.name,
        miniters=1
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                progress_bar.update(len(chunk))

    if expected_md5:
        if not verify_md5(output_path, expected_md5):
            raise ValueError(f"Downloaded file '{output_path.name}' failed MD5 verification!")
        logger.info(f"MD5 verified for '{output_path.name}'.")

    return output_path


@timeit
def download_tep_data(
    raw_dir: Path = DATA_RAW_DIR,
    file_keys: Optional[list] = None
) -> Dict[str, Path]:
    """
    Downloads raw .RData files for TEP dataset from Harvard Dataverse.
    
    Args:
        raw_dir: Destination directory for raw files.
        file_keys: List of keys from RAW_FILES to download. If None, downloads all files.
        
    Returns:
        Dictionary mapping file key to downloaded file Path.
    """
    keys_to_download = file_keys if file_keys else list(RAW_FILES.keys())
    downloaded_paths = {}

    for key in keys_to_download:
        if key not in RAW_FILES:
            raise KeyError(f"Unknown file key '{key}'. Available keys: {list(RAW_FILES.keys())}")

        file_meta = RAW_FILES[key]
        url = f"{DATAVERSE_BASE_URL}{file_meta['file_id']}"
        output_path = raw_dir / file_meta["filename"]

        downloaded_path = download_file(
            url=url,
            output_path=output_path,
            expected_md5=file_meta.get("md5"),
            expected_size=file_meta.get("size_bytes")
        )
        downloaded_paths[key] = downloaded_path

    return downloaded_paths


if __name__ == "__main__":
    download_tep_data()
