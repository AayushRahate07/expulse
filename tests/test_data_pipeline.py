"""
Unit Tests for Data Acquisition, Subsetting, Column Cleanup, and Format Conversion Pipeline.
"""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.config import MAX_SIMULATION_RUN, INCLUDED_FAULTS, EXCLUDED_VARIABLES
from src.data.convert import clean_column_names, process_single_rdata_file


def test_clean_column_names():
    """Test column name normalization and sanitization."""
    raw_cols = pd.Index(["faultNumber", "simulationRun", "XMV(5)", "XMV(9)", " xmeas_1 ", "XMEAS(10)"])
    cleaned = clean_column_names(raw_cols)
    assert cleaned == ["faultnumber", "simulationrun", "xmv_5", "xmv_9", "xmeas_1", "xmeas_10"]


def test_config_constants():
    """Verify configuration boundaries match proposal specification."""
    assert MAX_SIMULATION_RUN == 100
    assert INCLUDED_FAULTS == [0, 1, 4, 6, 8]
    assert "xmv_5" in EXCLUDED_VARIABLES or "XMV(5)" in EXCLUDED_VARIABLES
    assert "xmv_9" in EXCLUDED_VARIABLES or "XMV(9)" in EXCLUDED_VARIABLES


def test_synthetic_dataframe_subsetting():
    """Test dataframe filtering for simulationRun, faultNumber, and excluded variables."""
    data = {
        "faultNumber": [0, 0, 1, 2, 4, 6, 8],
        "simulationRun": [1, 105, 50, 10, 70, 80, 90],
        "sample": list(range(1, 8)),
        "XMV(5)": [1.0] * 7,
        "XMV(9)": [2.0] * 7,
        "xmeas_1": np.random.randn(7)
    }
    df = pd.DataFrame(data)

    # Filter simulationRun <= 100
    df_filtered = df[df["simulationRun"] <= MAX_SIMULATION_RUN]
    assert df_filtered["simulationRun"].max() <= 100
    assert 105 not in df_filtered["simulationRun"].values

    # Filter included faults [0, 1, 4, 6, 8]
    df_faults = df_filtered[df_filtered["faultNumber"].isin(INCLUDED_FAULTS)]
    assert 2 not in df_faults["faultNumber"].values

    # Drop excluded variables
    cols_to_drop = [c for c in df_faults.columns if c in EXCLUDED_VARIABLES]
    df_clean = df_faults.drop(columns=cols_to_drop)
    assert "XMV(5)" not in df_clean.columns
    assert "XMV(9)" not in df_clean.columns
