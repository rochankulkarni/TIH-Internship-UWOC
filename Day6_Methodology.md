# Day 6: Pre-Dataset Readiness + Methodology V1

## 📋 Deliverables Overview
* **Technical Methodology Document (Version 1.0)**
* **Modeling Strategy & Hyperparameter Configuration**
* **Mentor Presentation & Review Strategy**

---

## 🔬 Modeling Strategy & Evaluation Rigor

### 1. The Reference Baseline: Fixed-Threshold Detector
The hardware-emulated control baseline acts as a zero-intelligence comparator. Incoming signals are analyzed via a singular split threshold parameter:
$$\hat{y} = \begin{cases} 1, & \text{if } V_{\text{signal}} \ge \tau \\ 0, & \text{if } V_{\text{signal}} < \tau \end{cases}$$
* **Constraint:** Highly vulnerable to severe baseline drift and forward light scattering attenuation across murky NTU water settings.

### 2. Candidate ML Models & Hyperparameter Boundaries
To outpace the baseline detector under heavy turbidity variations, three model tracks are evaluated:

* **Logistic Regression (Linear Benchmark)**
  * *Purpose:* Establishes a lightweight, linearly bound classification boundary.
  * *Parameters:* Regularization penalties $L_1$ vs $L_2$; optimization tolerance factor set via $C \in [0.01, 0.1, 1.0, 10.0]$.

* **Support Vector Machines (Non-Linear Boundary Separation)**
  * *Purpose:* Maps multi-dimensional signal features to isolate optimal structural decision margins under noisy signal profiles.
  * *Parameters:* Kernel configurations evaluated over `linear`, `rbf` (Radial Basis Function); margin constraints measured across $C \in [0.1, 1, 10]$.

* **Random Forest Classifier (Ensemble Decision Architecture)**
  * *Purpose:* Addresses multi-variable interactions (e.g., variance, rise time, amplitude drops) across variable water conditions without over-indexing.
  * *Parameters:* Tree instances bounded to `n_estimators: [50, 100, 200]`; tree depth capped to `max_depth: [4, 8, 12]` to counter overfitting.

---

## 📉 Turbidity Performance Evaluation Protocol
Instead of averaging validation tests together, model performance curves are systematically isolated to reveal environmental breaking points:

