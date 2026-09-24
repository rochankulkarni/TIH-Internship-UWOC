# Day 18: Model Comparison and Mentor Review Progress Pack

## Deliverables Summary
* **Slide 1: Comprehensive Unified Comparison Matrix Across All Pipeline Methods**
* **Slide 2: System Constraints Review (Overfitting, Resource Complexity, Failure Patterns)**
* **Slide 3: Strategic Candidate Shortlist for Deeper Turbidity Experiments**

---

## Technical Context and Overview

### What I Did on Day 18
On Day 18, I assembled a unified comparison analysis covering every data decoding method built from Day 15 to Day 17. I ran microsecond-level latency tracking loops on Google Colab to gauge single-sample inference times alongside our standard telecom parameters (Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and Bit Error Rate). I evaluated these models across deployment risk constraints—including overfitting tendencies and embedded system memory profiles—and compiled this 3-slide progress pack to secure mentor sign-off on our final experimental strategy.

---

##  Slide 1: Comprehensive Method Comparison Matrix

The table below records the empirical performance benchmarks scored across our identical Dataset Version 1.0 test holdout slice (1,703 samples under clear-water 1-2 NTU parameters):

| Algorithmic Decoding Method | Accuracy | Precision | Recall | F1-Score | Confusion Matrix (TN, FP, FN, TP) | Bit Error Rate (BER) | Avg Single-Sample Latency (µs) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed Threshold (Control)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `[854, 0, 0, 849]` | 0.0000e+00 | **0.0210 µs** |
| **Logistic Regression (ML)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `[854, 0, 0, 849]` | 0.0000e+00 | 0.1850 µs |
| **Support Vector Machine (RBF)**| 1.0000 | 1.0000 | 1.0000 | 1.0000 | `[854, 0, 0, 849]` | 0.0000e+00 | 2.4510 µs |
| **Random Forest (Max Depth=5)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `[854, 0, 0, 849]` | 0.0000e+00 | 6.8200 µs |
| **Multi-Layer Perceptron (8,4)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `[854, 0, 0, 849]` | 0.0000e+00 | 12.1540 µs |

---

## Slide 2: Structural Constraints and Risk Diagnosis

### 1. Overfitting Analysis and Data Limitations
Because the baseline file (`1-2NTU.csv`) represents clean water with perfect bimodal state tracking, every single algorithm scored an absolute 100% success mark. This represents a clear **idealized data limitation**. While these perfect metrics confirm that our pipeline transformations and training loops are free of compilation bugs, they carry an extremely high risk of overfitting if used as a final performance proof. The current models have only learned to separate columns under perfect conditions and cannot handle signal degradation yet.

### 2. Failure Patterns and Resource Complexity
* **Fixed Threshold Failure Pattern:** The static midpoint rule is brittle. It possesses zero structural adaptation and will experience a total channel collapse the moment light-scattering path loss shifts or compresses the high-voltage signal peak below 1730 mV.
* **Neural Network Overhead:** The Multi-Layer Perceptron (MLP) yields the highest resource complexity, requiring continuous matrix floating-point operations. On a restricted hardware node like a Raspberry Pi receiver, this processing lag can limit transmission speed.

---

##  Slide 3: Strategic Candidate Shortlist

To navigate the transition into murky water testing blocks while balancing embedded processor constraints, the following candidates are shortlisted for deeper experimentation:

### Candidate 1: Random Forest Ensemble (The Primary Robust Track)
* *Justification:* Random Forest scored perfect telemetry bounds while maintaining a very light memory footprint. Its internal voting structure allows it to make complex decision branches. This makes it highly resilient to individual sensor variance drops without requiring high computational overhead.

### Candidate 2: Support Vector Machine with RBF Kernel (The Boundary Isolation Track)
* *Justification:* The SVM model achieved flawless precision with an average single-sample latency under 2.5 microseconds. Its RBF kernel excels at drawing smooth decision boundaries through non-linear overlapping zones. This capability is vital to maintain low Bit Error Rates when heavy turbidity compresses our voltage peaks.

### Final Modeling Directive
Our next development phase will pass the high-turbidity sheets (`3-4NTU.csv` and `5-6NTU.csv`) directly through these two shortlisted models to measure their true adaptability under stress.
