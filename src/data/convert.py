"""
Memory-Decoupled .RData to Parquet Data Preprocessing and Subsetting Pipeline.
"""

import gc
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pyreadr

from src.config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    RAW_FILES,
    MAX_SIMULATION_RUN,
    INCLUDED_FAULTS,
    EXCLUDED_VARIABLES,
    TRAIN_FAULT_FREE_RUNS,
    HOLDOUT_FAULT_FREE_RUNS,
)
from src.utils import logger, timeit


def clean_column_names(columns: pd.Index) -> List[str]:
    """Standardizes column names to clean snake_case strings."""
    cleaned = []
    for col in columns:
        c = str(col).strip()
        # Convert XMV(5) -> xmv_5 format or keep lowercase standard
        c_clean = c.lower().replace("(", "_").replace(")", "").replace(" ", "_")
        cleaned.append(c_clean)
    return cleaned


def process_single_rdata_file(
    key: str,
    raw_dir: Path = DATA_RAW_DIR,
    processed_dir: Path = DATA_PROCESSED_DIR,
    max_run: int = MAX_SIMULATION_RUN,
    included_faults: List[int] = INCLUDED_FAULTS,
    excluded_vars: List[str] = EXCLUDED_VARIABLES
) -> Path:
    """
    Reads a single .RData file, filters simulation runs & fault types,
    drops uninformative near-constant variables, converts to Parquet,
    and forces explicit garbage collection to prevent memory leaks.
    """
    file_meta = RAW_FILES[key]
    rdata_path = raw_dir / file_meta["filename"]
    parquet_path = processed_dir / f"{key}.parquet"

    if not rdata_path.exists():
        raise FileNotFoundError(f"Raw file '{rdata_path}' not found. Run downloader first.")

    logger.info(f"Loading '{rdata_path.name}' via pyreadr...")
    result = pyreadr.read_r(str(rdata_path))

    var_name = file_meta["r_var_name"]
    if var_name not in result:
        # Fallback if key differs
        var_name = list(result.keys())[0]

    df: pd.DataFrame = result[var_name]
    logger.info(f"Loaded '{key}' - Raw shape: {df.shape}")

    # Standardize column name lookup
    col_mapping = {col: str(col).strip() for col in df.columns}
    df.rename(columns=col_mapping, inplace=True)

    # 1. Filter simulationRun <= max_run
    if "simulationRun" in df.columns:
        df = df[df["simulationRun"] <= max_run]
    elif "simulationrun" in df.columns:
        df = df[df["simulationrun"] <= max_run]

    # 2. Filter faultNumber
    if "faultNumber" in df.columns:
        if "faultfree" in key.lower():
            df = df[df["faultNumber"] == 0]
        else:
            df = df[df["faultNumber"].isin(included_faults)]

    # 3. Drop excluded constant variables (XMV(5), XMV(9))
    cols_to_drop = [c for c in df.columns if c in excluded_vars or c.lower() in [e.lower() for e in excluded_vars]]
    if cols_to_drop:
        logger.info(f"Dropping constant variables: {cols_to_drop}")
        df.drop(columns=cols_to_drop, inplace=True)

    # 4. Standardize column headers
    df.columns = clean_column_names(df.columns)

    # 5. Convert integer types to memory-efficient dtypes
    for col in ["faultnumber", "simulationrun", "sample"]:
        if col in df.columns:
            df[col] = df[col].astype("int32")

    # Float columns to float32 for compact storage & speed
    float_cols = [c for c in df.columns if c not in ["faultnumber", "simulationrun", "sample"]]
    df[float_cols] = df[float_cols].astype("float32")

    # 6. For fault-free training/testing datasets, tag train/holdout split if needed
    if key == "fault_free_training":

        logger.info(f"Fault-free training set: {df['simulationrun'].nunique()} simulation runs.")
    elif key == "fault_free_testing":

        logger.info(f"Fault-free testing set: {df['simulationrun'].nunique()} simulation runs.")

    # 7. Write to compressed Parquet
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False, compression="snappy")
    logger.info(f"Saved processed '{key}' to '{parquet_path}' - Processed shape: {df.shape}")

    # Explicit memory cleanup (Refinement 2)
    del df, result
    gc.collect()
    logger.info(f"Memory cleared for '{key}'.")

    return parquet_path


@timeit
def process_and_convert_rdata_to_parquet(
    raw_dir: Path = DATA_RAW_DIR,
    processed_dir: Path = DATA_PROCESSED_DIR,
    file_keys: Optional[List[str]] = None
) -> Dict[str, Path]:
    """
    Sequentially processes and converts selected raw .RData files to Parquet.
    Each file is loaded, processed, written, and garbage-collected independently.
    """
    keys = file_keys if file_keys else list(RAW_FILES.keys())
    output_paths = {}

    for key in keys:
        logger.info(f"--- Processing [{key}] ---")
        output_paths[key] = process_single_rdata_file(
            key=key,
            raw_dir=raw_dir,
            processed_dir=processed_dir
        )

    return output_paths


if __name__ == "__main__":
    process_and_convert_rdata_to_parquet()
