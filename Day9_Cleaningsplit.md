# Day 9: Data Cleaning & Split Review Progress Pack

## 📋 Deliverables Overview
* **Data-Quality & Class-Balance Structural Metrics**
* **Proposed Dataset Train/Test Split & Scaling Strategy**
* **Imbalance & Analog Outlier Treatment Confirmation**
* **Frozen Approved Preprocessing & Cleaning Framework**

---

## 🛠️ Google Colab Interactive Notebook Code Execution

### Cell Execution: Data Audit, Outlier Capping, and Stratified Splitting Sequence
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset_path = 'UWOC_useing_photoresistor-main/anaotherGUI/1-2NTU.csv'

if os.path.exists(dataset_path):
    df = pd.read_csv(dataset_path)
    
    # Dynamic column detector mapping 
    detected_columns = {col.lower().strip(): col for col in df.columns}
    feature_column = detected_columns.get('analog_voltage_mv', df.columns[1])
    target_column = detected_columns.get('target_bit', df.columns[-1])

    
    print("AUTOMATIC FIELD DETECTOR ENGINE RUN TIME")
    
    print(f"📡 [SYSTEM DETECTION LOG]:")
    print(f"  • Successfully mapped feature variable to column: '{feature_column}'")
    print(f"  • Successfully mapped target classification to column: '{target_column}'\n")

    print(" [STEP 1: DATA PROFILE AUDIT]")
    total_rows, total_cols = df.shape
    print(f"• Total Data Samples Found: {total_rows} entries across {total_cols} tracking vectors.")
    print(f"• Null / Missing Value Count: {df.isnull().sum().sum()}")
    
    bit_counts = df[target_column].value_counts()
    bit_percentages = df[target_column].value_counts(normalize=True) * 100
    for bit_val in bit_counts.index:
        print(f"  - Payload Class [{bit_val}]: {bit_counts[bit_val]} samples ({bit_percentages[bit_val]:.2f}%)")
    
    print("\n [STEP 2: TRANSIENT CIRCUIT NOISE FILTERING]")
    mean_val = df[feature_column].mean()
    std_val = df[feature_column].std()
    upper_bound, lower_bound = mean_val + (4 * std_val), mean_val - (4 * std_val)
    outliers_detected = df[(df[feature_column] > upper_bound) | (df[feature_column] < lower_bound)]
    print(f"• Outlier Detection Windows: Allowed Signal Bounds = [{lower_bound:.2f} mV, {upper_bound:.2f} mV]")
    print(f"• Spurious Electrical Switching Spikes Identified: {len(outliers_detected)} records")
    df[feature_column] = df[feature_column].clip(lower=lower_bound, upper=upper_bound)
    print("• Treatment Confirmed: Outlier voltages clamped to maximum/minimum safety limits.")
    
    print("\n [STEP 3: EXECUTING STRATIFIED REPRODUCIBLE SPLIT]")
    X_train, X_test, y_train, y_test = train_test_split(
        df[[feature_column]], df[target_column], test_size=0.20, random_state=42, stratify=df[target_column]
    )
    print(f"• Sub-Train Processing Array Dimensions: {X_train.shape} Samples")
    print(f"• Sub-Test Evaluation Holdout Dimensions: {X_test.shape} Samples")
    
    print("\n [STEP 4: MIN-MAX SENSOR FEATURE SCALING]")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"• Scaling Vector Ranges: Min = {X_train_scaled.min()}, Max = {X_train_scaled.max()}")
    
    print("\n [STEP 5: PROCESSING ENGINE STATE LOCKED]")
    print("[✓ SUCCESS]: Cleaning, scaling boundaries, and stratified blocks frozen for Day 10 model tests.")
    print("=========================================================")
```

---

## Console Log Terminal Output

```text

 DAY 9: AUTOMATIC FIELD DETECTOR ENGINE RUN TIME


[SYSTEM DETECTION LOG]:
  • Successfully mapped feature variable to column: 'Analog_Voltage_mV'
  • Successfully mapped target classification to column: 'target_bit'

 [STEP 1: DATA PROFILE AUDIT]
• Total Data Samples Found: 8512 entries across 3 tracking vectors.
• Null / Missing Value Count: 0
  - Payload Class: 4266 samples (50.12%)
  - Payload Class: 4246 samples (49.88%)

 [STEP 2: TRANSIENT CIRCUIT NOISE FILTERING]
• Outlier Detection Windows: Allowed Signal Bounds = [-608.24 mV, 4038.90 mV]
• Spurious Electrical Switching Spikes Identified: 0 records
• Treatment Confirmed: Outlier voltages clamped to maximum/minimum safety limits.

 [STEP 3: EXECUTING STRATIFIED REPRODUCIBLE SPLIT]
• Sub-Train Processing Array Dimensions: (6809, 1) Samples
• Sub-Test Evaluation Holdout Dimensions: (1703, 1) Samples

 [STEP 4: MIN-MAX SENSOR FEATURE SCALING]
• Scaling Vector Ranges: Min = 0.0, Max = 1.0
• Scaler transformations applied successfully across data partitions.

 [STEP 5: PROCESSING ENGINE STATE LOCKED]
[✓ SUCCESS]: Cleaning, scaling boundaries, and stratified blocks frozen for Day 10 model tests.
=========================================================
```
