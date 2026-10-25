# Expulse: Multivariate Industrial Process Anomaly Detection on Tennessee Eastman Process (TEP)

An unsupervised anomaly detection pipeline and comparative benchmark evaluating three detection paradigms fit exclusively on nominal (fault-free) operating data from the Tennessee Eastman Process (TEP) benchmark dataset (Rieth et al., 2017).

---

## 1. Project Overview

Real-world industrial process units (reactors, condensers, separators, compressors, strippers) operate under dense cross-sensor correlation. Disturbances often manifest not as a single sensor crossing a hard threshold, but as joint multivariate shifts across connected units.

This implementation compares three anomaly detection approaches:
1. **Rule-Based Per-Sensor Thresholds**: Independent univariate $\mu \pm k\sigma$ tolerance bands.
2. **PCA Reconstruction Error**: Linear subspace projection measuring Squared Prediction Error (SPE).
3. **Isolation Forest**: Non-linear tree-based recursive partitioning.

Root-cause attribution is performed on flagged anomalous samples using **TreeSHAP** for Isolation Forest and **Per-Variable Squared Reconstruction Error** for PCA.

---

## 2. Repository Structure

```
expulse/
├── app.py                     # Streamlit interactive visualization dashboard
├── pyproject.toml             # Package configuration and build dependencies
├── requirements.txt           # Pinned Python package dependencies
├── README.md                  # Implementation documentation
├── data/
│   ├── raw/                   # Raw .RData files downloaded from Harvard Dataverse (ignored by git)
│   └── processed/             # Subsampled, sanitized Parquet files (ignored by git)
├── models/                    # Serialized model binaries (.joblib)
├── reports/
│   ├── evaluation_summary.csv              # Three-way comparative evaluation metrics across 100 runs
│   ├── idv4_pca_attribution.csv
│   ├── idv4_treeshap_attribution.csv
│   ├── cusum_vs_consecutive_leadtime.csv   # Page CUSUM vs heuristic delay benchmark
│   ├── attribution_accuracy_benchmark.csv  # Top-K hit rates and Wilcoxon test vs plant physics
│   ├── conformal_coverage_analysis.csv     # Non-overlapping block conformal false-alarm control
│   ├── temporal_features_benchmark.csv     # R^50 -> R^200 dynamic feature benchmark (IDV 8)
│   └── figures/                            # Generated 300 DPI publication plots
├── src/
│   ├── config.py              # Central constants, file IDs, fault boundaries, variables
│   ├── utils.py               # Seed setting, streaming MD5 validation, timing, logging
│   ├── data/
│   │   ├── downloader.py      # Automated download from Harvard Dataverse with MD5 checks
│   │   ├── convert.py         # Memory-decoupled .RData ingestion & Parquet conversion
│   │   └── features.py        # Rolling temporal augmentation (mean, volatility, trend slope)
│   ├── models/
│   │   ├── rule_based.py      # RuleBasedDetector (BaseEstimator)
│   │   ├── pca_detector.py    # PCADetector (BaseEstimator)
│   │   ├── isolation_forest.py# IsolationForestDetector (BaseEstimator)
│   │   └── consensus.py       # Tri-Level consensus advisory arbitrator (Levels 1-3)
│   ├── explainability/
│   │   ├── shap_explainer.py  # TreeSHAP computation & ranking
│   │   ├── pca_attribution.py # Per-variable reconstruction error attribution
│   │   └── validation.py      # Ground-truth mapping (Downs & Vogel) & Wilcoxon test
│   └── evaluation/
│       ├── metrics.py         # Precision, Recall, F1, FAR, 3-min lead time calculations
│       ├── cusum.py           # Page CUSUM sequential change-point detector (Lorden-optimal)
│       └── conformal.py       # Non-overlapping block conformal prediction for serial correlation
├── scripts/
│   ├── 01_download_data.py    # Download raw RData files
│   ├── 02_convert_parquet.py  # Convert raw files to Parquet
│   ├── 03_train_detectors.py  # Fit all three detectors on fault-free training data
│   ├── 04_evaluate.py         # Run three-way comparative benchmark on test data
│   ├── 05_explain.py          # Compute and rank feature attributions for IDV(4)
│   ├── 06_benchmark_cusum.py  # CUSUM vs heuristic windowing delay benchmark
│   ├── 07_validate_attributions.py # Top-K hit rate (H@1..5), MRR, and Wilcoxon validation
│   ├── 08_evaluate_block_conformal.py # Empirical block FAR evaluation on 96k healthy samples
│   ├── 09_benchmark_temporal_features.py # Dynamic R^200 feature expansion benchmark
│   └── generate_plots.py      # Generate publication-quality figures
├── tests/
│   ├── test_data_pipeline.py   # Data subsetting, sanitization, config tests
│   ├── test_models.py          # BaseEstimator API contract & shape tests
│   ├── test_metrics.py         # Classification metrics & lead-time unit tests
│   ├── test_explainability.py  # Attribution ranking and reconstruction error tests
│   ├── test_reproducibility.py # Determinism & seed reproducibility verification
│   └── run_tests.py            # Standalone zero-dependency test runner
```

---

## 3. Data Ingestion & Preprocessing

The primary dataset is the extended Tennessee Eastman Process simulation dataset published by Rieth et al. (2017) hosted on Harvard Dataverse (`DOI: 10.7910/DVN/6C3JR1`).

### Subsetting & Scope Constraints
- **Simulation Runs**: First 100 simulation runs (`simulationRun <= 100`) per condition (out of 500 available runs).
- **Fault Types**:
  - `IDV(0)`: Nominal (fault-free) operating conditions.
  - `IDV(1)`: A/C feed ratio step disturbance (large, clearly separable).
  - `IDV(4)`: Reactor cooling water inlet temperature step disturbance (subtler, thermal lag).
  - `IDV(6)`: A feed loss step disturbance (large step fault).
  - `IDV(8)`: A, B, C feed composition random variation disturbance.
- **Variable Pruning**: Dropped `XMV(5)` (compressor recycle valve) and `XMV(9)` (stripper steam valve), which maintain near-zero variance under nominal operating conditions. The remaining 50 variables (41 process measurements + 9 manipulated variables) are retained.
- **Memory Decoupling**: Each `.RData` file is ingested sequentially via `pyreadr`, filtered, converted to compressed Parquet (`snappy`), and purged using explicit `del` and `gc.collect()`.

### Dataset Footprint Reduction

| Dataset File | Raw Rows | Processed Rows | Raw Size (.RData) | Parquet Size |
| :--- | :--- | :--- | :--- | :--- |
| `TEP_FaultFree_Training.RData` | 250,000 | 50,000 | 23.5 MB | 4.2 MB |
| `TEP_FaultFree_Testing.RData` | 480,000 | 96,000 | 45.1 MB | 7.6 MB |
| `TEP_Faulty_Training.RData` | 5,000,000 | 200,000 | 471.2 MB | 16.8 MB |
| `TEP_Faulty_Testing.RData` | 9,600,000 | 384,000 | 798.1 MB | 30.5 MB |
| **Total** | **15,330,000** | **730,000** | **~1.34 GB** | **~59.1 MB** |

---

## 4. Detection Models

All three detectors inherit from `sklearn.base.BaseEstimator` and `sklearn.base.ClassifierMixin`, adhering to a standardized API:
- `.fit(X)`: Fits exclusively on nominal training runs and learns the healthy threshold.
- `.decision_function(X)`: Returns continuous anomaly scores where higher values denote greater anomaly severity.
- `.predict(X)`: Returns binary predictions (`0` for nominal, `1` for anomalous).

### 4.1 RuleBasedDetector (`src/models/rule_based.py`)
- **Assumption**: Every feature independently stays within an operating envelope.
- **Fit**: Fits per-feature mean $\mu_j$ and sample standard deviation $\sigma_j$ on nominal training data. Sets bounds $[\mu_j - k\sigma_j, \mu_j + k\sigma_j]$ with default $k=3.0$.
- **Score**: Maximum normalized deviation outside the boundary across all features:
  $$s(x) = \max_j \left( \max\left(0, \frac{x_j - (\mu_j + k\sigma_j)}{\sigma_j}, \frac{(\mu_j - k\sigma_j) - x_j}{\sigma_j}\right) \right)$$
- **Threshold**: Evaluated using the Tukey IQR rule ($Q_3 + 1.5 \times \text{IQR}$) over the nominal training score distribution.

### 4.2 PCADetector (`src/models/pca_detector.py`)
- **Assumption**: Nominal operational dynamics lie within a low-dimensional linear subspace.
- **Fit**: Scales data using `StandardScaler` and fits `PCA` to retain principal components explaining $\ge 95\%$ of cumulative variance (retained 35 components out of 50).
- **Score**: Squared Prediction Error (SPE / Reconstruction Error):
  $$\text{SPE}(x) = \|x_{\text{scaled}} - \hat{x}_{\text{scaled}}\|_2^2$$
  where $\hat{x}_{\text{scaled}} = V_k V_k^T x_{\text{scaled}}$.
- **Threshold**: $Q_3 + 1.5 \times \text{IQR}$ of the nominal training SPE distribution (learned threshold: `5.7526`).

### 4.3 IsolationForestDetector (`src/models/isolation_forest.py`)
- **Assumption**: Nominal points reside in dense clusters requiring deeper isolation paths; anomalies are isolated closer to tree roots.
- **Fit**: Scales data using `StandardScaler` and fits `sklearn.ensemble.IsolationForest` (100 estimators, random state 42).
- **Score**: Inverted path length score ($-\text{score\_samples}(x)$), aligning orientation so higher values correspond to greater anomaly severity.
- **Threshold**: $Q_3 + 1.5 \times \text{IQR}$ over nominal training scores (learned threshold: `0.4998`).

---

## 5. Explainability & Root-Cause Attribution

### 5.1 TreeSHAP (`src/explainability/shap_explainer.py`)
Applies `shap.TreeExplainer` directly to the fitted Isolation Forest ensemble. For anomalous sample vectors $X_{\text{anom}}$, feature importance is ranked by the mean absolute SHAP value:
$$\overline{|\phi_j|} = \frac{1}{M}\sum_{i=1}^M |\phi_j(x^{(i)})|$$

### 5.2 PCA Per-Variable Reconstruction Contribution (`src/explainability/pca_attribution.py`)
Computes the squared error contribution per individual sensor:
$$e_j(x) = (x_{\text{scaled}, j} - \hat{x}_{\text{scaled}, j})^2$$
Features are ranked by mean reconstruction error across flagged anomalous samples.

### Empirical Attribution Results for IDV(4) (Cooling Water Disturbance)
For `IDV(4)` (step change in reactor cooling water inlet temperature), both attribution techniques identify the primary physical process variables:

| Rank | Isolation Forest Top TreeSHAP Features | Mean \|SHAP\| | PCA Top Reconstruction Features | Mean Squared Error |
| :---: | :--- | :--- | :--- | :--- |
| 1 | **`xmv_10`** (Reactor cooling water valve) | 0.3457 | **`xmv_10`** (Reactor cooling water valve) | 13.5964 |
| 2 | `xmeas_34` (Stripper pressure) | 0.0837 | **`xmeas_9`** (Reactor temperature) | 11.6466 |
| 3 | **`xmeas_9`** (Reactor temperature) | 0.0830 | `xmeas_31` (Separator pressure) | 0.3575 |
| 4 | `xmeas_22` (Separator cooling water temp)| 0.0783 | `xmeas_38` (Stripper steam) | 0.3374 |
| 5 | `xmeas_12` (Separator level) | 0.0774 | `xmeas_29` (Separator temperature) | 0.2283 |

---

## 6. Evaluation Protocol & Results

- **Sampling Frequency**: 3 minutes per sample (Rieth et al., 2017).
- **Fault Introduction**: Sample 160 (hour 8.0) in all testing runs.
- **Lead Time Metric**:
  $$\text{Lead Time (samples)} = \text{Sample}_{\text{first sustained flag}} - 160$$
  $$\text{Lead Time (minutes)} = \text{Lead Time (samples)} \times 3$$
  where a sustained flag requires $N=3$ consecutive predictions above threshold.
- **Evaluation Set**: 100 simulation runs per fault type ($400$ total simulation runs, $384,000$ test observations).

### Three-Way Comparative Benchmark

| Fault Type | Model | Precision | Recall | F1-Score | Lead Time (samples) | Lead Time (minutes) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **IDV(1)** | **Rule-Based** | 1.000 | 0.993 | **0.996** | +5.6 | +16.7 min |
| | **PCA Reconstruction** | 0.996 | 0.991 | 0.993 | +1.1 | **+3.3 min** |
| | **Isolation Forest** | 0.999 | 0.736 | 0.846 | +7.9 | +23.7 min |
| **IDV(4)** | **Rule-Based** | 1.000 | 0.998 | **0.999** | +1.1 | +3.4 min |
| | **PCA Reconstruction** | 0.996 | 0.998 | 0.997 | -0.1 | **-0.2 min (Immediate)** |
| | **Isolation Forest** | 0.978 | 0.032 | 0.061 | +274.6 | +823.8 min |
| **IDV(6)** | **Rule-Based** | 1.000 | 0.999 | **0.999** | +1.0 | +3.0 min |
| | **PCA Reconstruction** | 0.996 | 0.999 | 0.997 | -0.3 | **-1.1 min (Immediate)** |
| | **Isolation Forest** | 0.999 | 0.990 | 0.995 | +8.3 | +24.9 min |
| **IDV(8)** | **Rule-Based** | 1.000 | 0.961 | **0.980** | +25.4 | +76.3 min |
| | **PCA Reconstruction** | 0.995 | 0.922 | 0.957 | +18.5 | **+55.6 min** |
| | **Isolation Forest** | 0.999 | 0.958 | 0.978 | +26.4 | +79.3 min |

---

## 7. Execution Guide

### Installation
Ensure Python $\ge 3.10$ is installed:
```powershell
python -m pip install -r requirements.txt
```

### Reproducing the Pipeline Step-by-Step
```powershell
# Step 1: Download raw .RData files from Harvard Dataverse
python scripts/01_download_data.py

# Step 2: Convert .RData to compact Parquet files
python scripts/02_convert_parquet.py

# Step 3: Train Rule-Based, PCA, and Isolation Forest models
python scripts/03_train_detectors.py

# Step 4: Run three-way comparative evaluation across all 100 runs
python scripts/04_evaluate.py

# Step 5: Generate TreeSHAP and PCA feature attributions for IDV(4)
python scripts/05_explain.py

# Step 6: Generate publication figures in reports/figures/
python scripts/generate_plots.py
```

### Running the Interactive Results Viewer
Launch the local Streamlit dashboard:
`powershell
streamlit run app.py
`

### Running Unit Tests
Execute the test suite (12 automated unit, explainability & reproducibility tests):
`powershell
# Using the built-in runner (no external dependencies required):
python tests/run_tests.py

# Or via standard unittest / pytest:
python -m unittest
pytest tests/
`
---

## 8. Explicit Scope Boundaries (Implemented vs. Omitted)

| Component | Implemented in Current System | Explicitly Excluded / Omitted |
| :--- | :--- | :--- |
| **Dataset Scope** | 4 Fault types (`IDV 1, 4, 6, 8`) + Nominal baseline | All 20 TEP fault types |
| **Task Definition** | Binary Anomaly Detection & Lead Time analysis | Multi-class Fault Diagnosis (identifying fault ID) |
| **Data Partitioning** | 100 simulation runs per condition (70 train / 30 holdout nominal) | Full 500-run simulation dataset |
| **Feature Selection** | 50 continuous process/manipulated variables | Raw unpruned 52 variables (`XMV 5` & `XMV 9` dropped) |
| **Model Families** | Rule-Based thresholds, PCA Reconstruction, Isolation Forest | Deep learning models (LSTM Autoencoders, Transformers) |
| **Attribution** | TreeSHAP (Isolation Forest) & Squared Reconstruction Error (PCA) | Integrated Gradients, DeepLIFT, LIME |
| **Demonstration** | Precomputed Parquet-backed Streamlit dashboard | Live hardware/IoT/Arduino components |

---

## 9. Future Scope & Research Extensions

1. **Multi-Class Fault Diagnosis & Classification**:
   - Extend the pipeline from binary detection ($0$ vs $1$) to multi-class fault classification to diagnose *which* specific fault occurred after initial detection.
2. **Expansion to All 20 TEP Disturbance Modes**:
   - Evaluate detectors on difficult disturbances such as `IDV(3)` (step change in D feed temperature) and `IDV(9)` (random variation in D feed temperature) where closed-loop controllers often mask abnormal behavior.
3. **Deep Learning Temporal Baselines**:
   - Benchmark against sequence-aware reconstructors including **LSTM Autoencoders (LSTM-AE)**, **Temporal Convolutional Networks (TCN)**, and **Masked Autoencoder Transformers** to evaluate whether temporal autocorrelation modeling provides earlier warning lead time than static PCA.
4. **Dynamic & Adaptive Thresholding**:
   - Replace static IQR thresholding with time-varying dynamic limits (e.g., Exponentially Weighted Moving Average (EWMA), Cumulative Sum (CUSUM), or dynamic PCA with time lags) to better capture transient transitions.
5. **Real-Time Streaming Architecture**:
   - Adapt `src/models/` for sliding-window streaming inference via Apache Kafka or MQTT, simulating direct SCADA sensor telemetry integration.
