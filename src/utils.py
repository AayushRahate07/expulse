"""
Shared Utilities: Logging, Reproducibility Seeding, File Checksum Verification, and Performance Timers.
"""

import hashlib
import logging
import random
import time
from functools import wraps
from pathlib import Path
from typing import Callable, Any, Optional

import numpy as np


def setup_logger(name: str = "expulse", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = setup_logger("expulse")


def set_seed(seed: int = 42) -> None:
    """Sets global random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Global random seed set to {seed}")


def verify_md5(file_path: Path, expected_md5: str, chunk_size: int = 8192) -> bool:
    """Verifies MD5 checksum of a file against expected value."""
    if not file_path.exists():
        return False
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5_hash.update(chunk)
    actual_md5 = md5_hash.hexdigest().lower()
    matches = actual_md5 == expected_md5.lower()
    if not matches:
        logger.warning(
            f"MD5 mismatch for {file_path.name}: Expected {expected_md5}, got {actual_md5}"
        )
    return matches


def timeit(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to measure and log execution duration of a function."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start_time
        logger.info(f"Completed '{func.__name__}' in {duration:.2f} seconds")
        return result
    return wrapper
