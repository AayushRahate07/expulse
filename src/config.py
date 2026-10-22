"""
Global Configuration & Constants for Tennessee Eastman Process (TEP) Anomaly Detection.
"""

from pathlib import Path

# Directory Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure essential output directories exist
for path in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Harvard Dataverse API Source Specifications
DATAVERSE_BASE_URL = "https://dataverse.harvard.edu/api/access/datafile/"
DATAVERSE_DOI = "doi:10.7910/DVN/6C3JR1"

RAW_FILES = {
    "fault_free_training": {
        "file_id": 3031241,
        "filename": "TEP_FaultFree_Training.RData",
        "md5": "ec126484534331f85001d8c4ebce6d17",
        "size_bytes": 24678017,
        "r_var_name": "fault_free_training"
    },
    "fault_free_testing": {
        "file_id": 3031240,
        "filename": "TEP_FaultFree_Testing.RData",
        "md5": "38ad9810fc871026157086ae2c2f0ee9",
        "size_bytes": 47327663,
        "r_var_name": "fault_free_testing"
    },
    "faulty_training": {
        "file_id": 3031242,
        "filename": "TEP_Faulty_Training.RData",
        "md5": "c5f594d54c47e620ff877feb58407fda",
        "size_bytes": 494063194,
        "r_var_name": "faulty_training"
    },
    "faulty_testing": {
        "file_id": 3031243,
        "filename": "TEP_Faulty_Testing.RData",
        "md5": "556bdb64c83021bc0c5f92e427753565",
        "size_bytes": 836882037,
        "r_var_name": "faulty_testing"
    }
}

# Subsetting Boundaries (Proposal Section 3.4)
MAX_SIMULATION_RUN = 100
TRAIN_FAULT_FREE_RUNS = 70
HOLDOUT_FAULT_FREE_RUNS = 30
INCLUDED_FAULTS = [0, 1, 4, 6, 8]

# Near-constant variables documented in TEP literature to drop
EXCLUDED_VARIABLES = ["xmv_5", "xmv_9", "XMV(5)", "XMV(9)"]

# Time & Sampling Parameters
SAMPLING_INTERVAL_MINUTES = 3
FAULT_INTRO_SAMPLE_TESTING = 160  # Fault introduced at sample 160 (8 hours in)
FAULT_INTRO_SAMPLE_TRAINING = 20  # Fault introduced at sample 20 (1 hour in) for training runs

# Default Random Seed
RANDOM_SEED = 42
