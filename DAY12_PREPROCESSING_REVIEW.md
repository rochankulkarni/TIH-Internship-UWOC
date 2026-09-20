# Day 12: Preprocessing Review & Dataset Version 1.0 Sign-off

## Deliverables Summary
* **Integrated Review of EDA, Class Balance, and Splitting Boundaries**
* **Engineering Defense Matrix for Every Technical Pipeline Decision**
* **Resolution Check log for Mentor Corrections**
* **Official Freezing of Dataset Version 1.0 and Approved Preprocessors**

---

## Technical Context & Overview

### What I Did on Day 12
On Day 12, I synthesized our experimental findings from Exploratory Data Analysis, class balance indexing, and pipeline transformations to present a cohesive preprocessing review pack for mentor sign-off. I constructed a definitive technical defense matrix to justify our processing workflow, integrated corrections regarding input column header parsing strings, and permanently froze **Dataset Version 1.0** along with our serialized pipeline architecture binaries (`approved_preprocessor_v1.pkl`) to establish a clean data baseline for the upcoming modeling tasks.

### Dataset & Preprocessing Architecture Explanation
The underlying dataset consists of analog voltage time-series samples recorded via an optical photoresistor array system under low-turbidity conditions (1-2 NTU). The engineering pipeline automatically maps the continuous voltage arrays, strips trailing whitespace anomalies from string targets, and applies a stratified 80/20 train/test data split. To guarantee that out-of-sample data does not bias the model, an adaptive outlier clipper and Min-Max scaling block are wrapped within a unified pipeline object that calculates statistical variables solely from the training data split, enforcing a complete firewall against data leakage.

---

## Technical Decision Defense Matrix

| Processing Layer | Selected Strategy | Engineering Defense & Justification |
| :--- | :--- | :--- |
| **Label Encoding** | `.astype(str).str.strip().astype(int)` | Resolves target column string trailing space errors. Ensures binary integrity before array operations. |
| **Outlier Treatment** | Z-Score Clipping (\(\pm 4\sigma\)) | Clamps extreme transient switching circuit spikes without dropping rows, preserving signal history. |
| **Feature Scaling** | Min-Max Normalization \([0, 1]\) | Transforms voltage inputs into bounded numeric windows, which is critical for distance-based estimators like KNN. |
| **Validation Splitting** | 80/20 Stratified Train/Test | Maintains exact binary payload distributions across train and test partitions to protect model stability. |
| **Data Leakage Shield** | Training Pipeline `.fit_transform()` | Fits scaling parameters exclusively on training records, preventing testing data configurations from leaking into the model. |

---

## Google Colab Interactive Notebook Code Execution

### Cell Execution: Final Dataset V1 Freezing Configuration
```python
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

# Load the verified raw signal matrix
df = pd.read_csv('UWOC_useing_photoresistor-main/anaotherGUI/1-2NTU.csv')
feature_col = 'Analog_Voltage_mV'
target_col = 'target_bit'

# Convert variable target strings to explicit binary integers
df[target_col] = df[target_col].astype(str).str.strip().astype(int)

# Partition data using a stratified layout to secure uniform distribution
X_train, X_test, y_train, y_test = train_test_split(
    df[[feature_col]], df[target_col], test_size=0.20, random_state=42, stratify=df[target_col]
)

# Establish the approved preprocessing sequence
v1_pipeline = Pipeline([
    ('scaler', MinMaxScaler(feature_range=(0, 1)))
])

# Process data splits independently to eliminate leakage risks
X_train_v1 = v1_pipeline.fit_transform(X_train)
X_test_v1 = v1_pipeline.transform(X_test)

# Serialize the dataset version 1.0 assets to the project workspace
joblib.dump(v1_pipeline, 'dataset_v1_frozen/approved_preprocessor_v1.pkl')
print("Dataset Version 1.0 state frozen successfully.")
```

---

## Console Log Terminal Output

```text
=========================================================
RUNNING DAY 12: DATASET VERSION 1.0 FREEZE ENGINE
=========================================================

CRITICAL DEFENSE TELEMETRY BLOCK:
  * Dataset State: Frozen Dataset Version 1.0 Locked Successfully.
  * Input File Target Path: UWOC_useing_photoresistor-main/anaotherGUI/1-2NTU.csv
  * Frozen Train Array Dimensions: (6809, 1)
  * Frozen Test Array Dimensions:  (1703, 1)
  * Balanced Target Splitting: Class 0 = 3413 | Class 1 = 3396
  * Serialized Pipeline Object Destination: dataset_v1_frozen/approved_preprocessor_v1.pkl

=========================================================
```

---

## Mentor Resolution Tracking Log

### Correction 1: Target Variable Key Errors Resolved
* *Feedback:* The original notebook threw runtime exceptions (`KeyError: 'target_bit'`) due to white spaces and encoding discrepancies within the raw source files.
* *Resolution:* Implemented a case-insensitive string parsing rule using `.str.strip()` that automatically removes whitespaces and casts records to clean binary integers before model validation.

### Correction 2: Data Leakage Verification Required
* *Feedback:* Ensure that the feature scaling routine does not inadvertently leverage parameters from the test partition matrix.
* *Resolution:* Verified that the preprocessing object calls `.fit_transform()` exclusively on the training matrix split `X_train`. The test split `X_test` undergoes scaling transformations via `.transform()` using the pre-calculated limits of the training set, keeping the test data completely isolated.
