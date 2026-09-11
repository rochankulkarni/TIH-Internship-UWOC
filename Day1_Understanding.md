# Day 1 — Problem Understanding & Research

## Objective & Problem Statement

To communicate underwater, we can't use WiFi or radio waves, because water absorbs electromagnetic waves quickly, giving them very short range underwater. So instead, this project uses **light** to send data underwater - bits are transmitted as light pulses light on = 1, light off = 0.

The main problem: impurities in water - measured by **turbidity, in NTU (Nephelometric Turbidity Units)** - scatter and absorb the light. Because of this, the receiver doesn't get a clean "on/off" signal. Instead, it gets a **wobbly analog number** e.g. 0.3, 0.6, 0.9 and has to guess which bit was actually sent.

The naive approach is to pick one fixed cutoff number say, 0.5 - anything above is classified as "1", anything below as "0". But as the water gets murkier (higher NTU), this fixed cutoff stops working well, because the signal shifts and gets noisier, and the "0" and "1" signal ranges start to overlap.

**Goal:** Build machine learning models that can look at the received signal value and the turbidity level, where available and correctly classify whether the original bit was 0 or 1 - even as turbidity changes -and show that they outperform the simple fixed-cutoff method.

## Requirements

- A labelled dataset containing received signal values, transmitted bit labels, and turbidity (NTU) information.
- Software-only implementation - no physical hardware setup or data collection required.
- Tools: Python, Jupyter/Google Colab, NumPy, Pandas, Matplotlib, Scikit-learn.
- Deliverables: literature review, EDA, baseline + ML model comparison, robustness analysis, final report and presentation.

## Dataset Understanding

The dataset provided is from an existing UWOC (Underwater Wireless Optical Communication) project built using a Raspberry Pi and a photoresistor-based receiver.

The main usable file is `anaotherGUI/1-2NTU.csv`, containing:
- `labels` - the true transmitted bit (0 or 1)
- `feature` - the received analog signal value
- `bit` - an existing threshold-based prediction of the bit

This file has ~8,513 rows, but covers only the **1–2 NTU turbidity range**. Other turbidity ranges (2–3, 3–4, 4–5, 5–6 NTU) exist in the project folders only as result plot images, not as raw data - this is a current limitation to flag, since the objective of comparing performance across multiple NTU levels can't be fully done yet with real numbers.

A second file, `Graph/ALL/output.csv` (400 rows), has similar information: `Binary Sent` (true bit), `Analog` (normalized signal value), and `Binary Bit R` (a received/predicted bit).

## Technical Concepts

- **Turbidity / NTU**: a measure of water cloudiness caused by suspended particles; higher NTU means murkier water and more light scattering/absorption.
- **Fixed threshold**: a single cutoff value used to classify a continuous signal into 0 or 1; simple but breaks down as noise/turbidity increases.
- **Logistic Regression**: a linear model that estimates the probability of a bit being 1 based on the signal value.
- **SVM (Support Vector Machine)**: finds the best boundary that separates 0s and 1s, even when the classes aren't perfectly separable.
- **Random Forest**: an ensemble of decision trees that can capture non-linear patterns in how signal value relates to the true bit.

## Overall Workflow

1. Inspect and clean the dataset.
2. Explore the data visually (e.g. distribution of `feature` values for label 0 vs label 1).
3. Build a fixed-threshold baseline and measure its accuracy/F1/BER.
4. Train ML models (Logistic Regression, SVM, Random Forest) on the same data.
5. Compare ML models against the baseline using accuracy, precision, recall, F1, confusion matrix, and BER.
6. Analyze robustness and failure cases, including the current limitation of only having raw data for one NTU range.
7. Document findings and prepare for mentor review.
