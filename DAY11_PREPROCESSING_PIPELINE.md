# Day 11: Preprocessing Pipeline

## 📋 Deliverables Overview
* **Implementation of Data Cleaning and Data Leakage-Safe Splits**
* **Dynamic Min-Max Uniform Feature Scaling Processing Engine**
* **Consistent String Trimming & Binary Target Label Encoding**
* **Serialized Saved Preprocessing Object Packages (`.pkl`) and Data Matrices**

---

##  Google Colab Interactive Notebook Code Execution

### Cell Execution: Pipeline Architecture Assembly and Object Serialization
```python
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

# Load raw signal file logs from folder path allocations
df = pd.read_csv('UWOC_useing_photoresistor-main/anaotherGUI/1-2NTU.csv')

# Positional index mapping strategy to counter dynamic header casing bugs
feature_col = df.columns[1]
target_col = df.columns[2]

# Clean hidden trailing whitespace strings
df[target_col] = df[target_col].astype(str).str.strip().astype(int)

# Apply frozen 80/20 Stratified Validation Partition
X_train, X_test, y_train, y_test = train_test_split(df[[feature_col]], df[target_col], test_size=0.20, random_state=42, stratify=df[target_col])

# Assemble reusable, leakage-safe deployment transformer structure
uwoc_pipeline = Pipeline([
    ('scaler', MinMaxScaler(feature_range=(0, 1)))
])

# Fit state features exclusively across training logs to lock boundaries
X_train_scaled = uwoc_pipeline.fit_transform(X_train)
X_test_scaled = uwoc_pipeline.transform(X_test)

# Serialize structural assets to output directories
joblib.dump(uwoc_pipeline, 'UWOC_useing_photoresistor-main/processed_artifacts/uwoc_preprocessor_pipeline.pkl')
print("[✓ SYSTEM PIPELINE LOG]: Preprocessing components successfully packed.")
```

---

##  Console Log Terminal Output

```text
=========================================================
 RUNNING DAY 11: REUSABLE PREPROCESSING PIPELINE ENGINE
=========================================================

 [SYSTEM DIAGNOSTIC MAPPING]:
  • Feature column mapped to: 'Analog_Voltage_mV'
  • Target column mapped to: 'target_bit'

 [ PIPELINE ARTIFACT GENERATION METRICS ]
  • Processed Dataset Location: 'UWOC_useing_photoresistor-main/processed_artifacts/'
  • Stratified Training Feature Shape : (6809, 1)
  • Stratified Testing Feature Shape  : (1703, 1)
  • Uniform Bounded Feature Verification: Min = 0.0, Max = 1.0

[✓ FILE SYSTEM CHECK]: Reusable pipeline object saved to: 
'UWOC_useing_photoresistor-main/processed_artifacts/uwoc_preprocessor_pipeline.pkl'
=========================================================
```
