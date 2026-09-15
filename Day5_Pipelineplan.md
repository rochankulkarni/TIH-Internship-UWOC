# Day 5: Pipeline & Experiment Planning

## 📋 Deliverables Overview
* **Pipeline Flowchart Blueprint**
* **Reproducible Workflow Architecture**
* **Experiment Log Template**
* **Project Directory Structure**

---

## 🛠️ Reproducible Workflow & Architecture
To ensure an end-to-end reproducible workflow from raw underwater optical communication (UWOC) signal data to final bit classification metrics, the data pipeline is organized into five isolated stages:

### 1. Ingestion & Quality Controls
* Validate incoming file signatures (`.csv`, `.parquet`, or `.mat`).
* Enforce schema structure: strict checking of expected datatypes, signal bounds, and sequence length integrity.
* Archive immutable raw files to a read-only root directory before any transformation.

### 2. Signal Preprocessing
* Apply noise-reduction filtering routines (e.g., low-pass filtering, rolling averaging) tailored to remove scattering noise.
* Normalize signal amplitude arrays uniformly across all variable turbidity runs.
* Handle missing data frames using fixed temporal interpolation constraints.

### 3. Feature Engineering & Windowing
* Segment continuous signal streams into discrete, single-bit time intervals.
* Extract statistical time-domain components: peak voltage, mean energy, variance, rise time, and fall time.
* Format the structured baseline arrays mapping target binary vectors `[0, 1]`.

### 4. Stratified Modeling Sandbox
* **Baseline Detector:** Implement a static, fixed-threshold decision boundary framework.
* **ML Classifiers:** Setup pipeline runners for Logistic Regression, Support Vector Machines (SVM), and Random Forests.
* **Validation Strategy:** Stratify folds strictly across discrete NTU turbidity parameters rather than pooling variables.

### 5. Evaluation Engine
* Compute primary classification performance metrics: Accuracy, F1-Score, and Bit Error Rate (BER).
* Generate complexity score indices profiling inference latency overheads per model block.

---

## 📊 Standardized Experiment Log Template
Every evaluation run must be logged using the following markdown index structure to preserve history:

| Run ID | Timestamp | Classifier Method | Hyperparameters Selected | Turbidity Level (NTU) | Accuracy | F1-Score | Bit Error Rate (BER) | Inference Latency | Artifact Path |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `RUN_001` | `2026-09-15 10:00` | Baseline Threshold | `thresh=0.45` | Pooled (All) | 0.8240 | 0.8110 | 1.76 × 10⁻¹ | 0.04 ms | `/models/base/` |
| `RUN_002` | `2026-09-15 10:45` | Random Forest | `n_est=100, max_d=8` | 10 NTU | 0.9850 | 0.9845 | 1.50 × 10⁻² | 2.15 ms | `/models/rf_v1/` |
| `RUN_003` | `2026-09-15 11:30` | Random Forest | `n_est=100, max_d=8` | 50 NTU | 0.8910 | 0.8870 | 1.09 × 10⁻¹ | 2.18 ms | `/models/rf_v1/` |

## 📁 Standardized Project Directory Structure
Maintain this repository configuration layout to enable automated execution dependencies across all local clusters or cloud runtimes:

```text
uwoc-bit-classification/
├── README.md               # Main repository documentation & setup rules
├── data/
│   ├── 01_raw/             # Read-only untouched source datasets
│   ├── 02_processed/       # Cleaned, standardized signal sequences
│   └── 03_features/        # Engineered training matrices (grouped by NTU)
├── notebooks/
│   ├── 1.0_eda.ipynb       # Exploratory analysis & signal inspection
│   └── 2.0_metrics.ipynb   # Plotting baseline error degradation curves
├── src/
│   ├── __init__.py
│   ├── ingestion.py        # Validations, parsing, and pipeline staging
│   ├── preprocessing.py    # Noise removal filters and normalization
│   ├── features.py         # Statistical feature extraction algorithms
│   └── models.py           # Training loops, cross-validation, and inference
├── metrics/
│   ├── experiment_log.md   # Tabular index of all performance histories
│   └── evaluation_plots/   # Exported confusion matrices & BER curves
├── config/
│   └── params.yaml         # Centralized hyperparameters and threshold boundaries
└── requirements.txt        # Exact environment library pins (sklearn, pandas, etc.)
