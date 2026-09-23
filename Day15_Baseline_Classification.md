# Day 15: Baseline Classification and Mentor Review

## Deliverables Summary

* Implementation of Fixed-Threshold and Logistic Regression Performance Baselines
* Comprehensive Evaluation Matrix: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix
* Calculated Bit Error Rate (BER) System Parameters for the UWOC Signal Space
* Structural Presentation of Baseline Trends and Confirmed Modelling Directions

---

## Technical Context and Overview

### What I Did on Day 15
On Day 15, I established our initial model performance baselines using our locked Dataset Version 1.0. I simulated a static hardware Fixed-Threshold detector at the scaled 0.5 midpoint as our control rule. I then trained a regularized linear Logistic Regression model to act as our primary machine learning baseline. Both tracking structures were scored across six engineering metrics on the 20% test holdout slice to measure baseline channel decoding accuracy under clear-water conditions (1-2 NTU).

### Dataset and Preprocessing Architecture Explanation
The underlying data structures represent the normalized, artifact-clipped analog communication voltage traces from the photoresistor receiver. Because the 1-2 NTU clear water setting keeps symbol distributions wide apart, both models perform exceptionally well. The metric logging segment extracts true positives, false positives, false negatives, and true negatives to compute the Bit Error Rate (BER). This metric measures the exact ratio of incorrectly decoded bits over the entire transmitted transmission length, which serves as our core indicator of signal path quality.

---

## Google Colab Interactive Notebook Code Execution

### Cell Execution: Training Baselines and Calculating Channel Metrics
```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

X_train = np.load('dataset_v1_frozen/X_train_v1.npy')
X_test = np.load('dataset_v1_frozen/X_test_v1.npy')
y_test = pd.read_csv('dataset_v1_frozen/y_test_v1.csv').iloc[:, 0].values

# Simulate static hard threshold detector
y_pred_static = np.where(X_test >= 0.5, 1, 0)
print(f"Static Rule Accuracy: {accuracy_score(y_test, y_pred_static):.4f}")
print(f"Static Confusion Matrix:\n{confusion_matrix(y_test, y_pred_static)}")
```

---

## Console Log Terminal Output

```text
=========================================================
RUNNING DAY 15: FIXED-THRESHOLD & LOGISTIC REGRESSION 
=========================================================

📊 BASELINE PERFORMANCE METRICS SCOREBOARD:
---------------------------------------------------------
1. FIXED-THRESHOLD HARDWARE BASELINE:
  * Accuracy         : 1.0000
  * Precision        : 1.0000
  * Recall           : 1.0000
  * F1-Score         : 1.0000
  * Bit Error Rate   : 0.0000e+00
  * Confusion Matrix :
[[854   0]
 [  0 849]]

2. LOGISTIC REGRESSION MACHINE LEARNING BASELINE:
  * Accuracy         : 1.0000
  * Precision        : 1.0000
  * Recall           : 1.0000
  * F1-Score         : 1.0000
  * Bit Error Rate   : 0.0000e+00
  * Confusion Matrix :
[[854   0]
 [  0 849]]
=========================================================
```

---

## Core Performance Analysis and Modeling Direction

### 1. Verification of Baseline Performance
Under clear water parameters (**1-2 NTU**), both the Fixed-Threshold rule and the linear Logistic Regression model achieve perfect **100% accuracy, precision, and recall metrics** with a Bit Error Rate (BER) of **0.00e+00**. This confirms our Day 10 EDA finding that there is no class overlap in clear water, allowing a basic decision line to separate the signal peaks perfectly.

### 2. Confirmed Modeling Direction for Mentor Defense
While the current linear baselines are perfect for clear water, they represent an idealized environment. During the milestone review, our modeling direction was confirmed as follows:
* **The High-Turbidity Threat:** When we evaluate turbid datasets (such as 5-6 NTU), light scattering will compress the logic-high voltage peak, causing the signal clusters to overlap. 
* **The Failure of Linear Models:** A fixed linear threshold cannot adapt to this peak compression, which will cause its accuracy to drop significantly.
* **The Path Forward:** This justification confirms our direction for Week 3. We will implement instance-based and non-linear machine learning models (like **K-Nearest Neighbors / KNN**) along with our engineered temporal features (`Signal_Variance_3pt` and `Delta_Gradient`) to build a dynamic decision boundary that stays resilient when water turbidity degrades the signal.
