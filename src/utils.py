"""
Shared Utilities: Logging, Reproducibility Seeding, File Checksum Verification, Table Formatting, and Performance Timers.
"""

import hashlib
import logging
import os
import random
import time
from functools import wraps
from pathlib import Path
from typing import Callable, Any, Optional

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED


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


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Sets global random seed for reproducibility across random, numpy, and Python hash seed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
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


def format_dataframe_as_markdown(df: pd.DataFrame, index: bool = False) -> str:
    """
    Renders a pandas DataFrame as a Markdown table.
    Uses pandas.to_markdown() if tabulate is installed, otherwise falls back
    to a pure-Python Markdown table formatter so CLI scripts never crash.
    """
    try:
        return df.to_markdown(index=index)
    except (ImportError, Exception):
        # Pure-Python Markdown table formatting fallback
        df_display = df.copy()
        if not index:
            headers = [str(c) for c in df_display.columns]
            rows = [[str(v) for v in row] for row in df_display.values]
        else:
            headers = ["Index"] + [str(c) for c in df_display.columns]
            rows = [[str(idx)] + [str(v) for v in row] for idx, row in zip(df_display.index, df_display.values)]

        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, val in enumerate(row):
                col_widths[i] = max(col_widths[i], len(val))

        header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
        sep_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
        row_lines = ["| " + " | ".join(val.ljust(col_widths[i]) for i, val in enumerate(row)) + " |" for row in rows]
        return "\n".join([header_line, sep_line] + row_lines)
