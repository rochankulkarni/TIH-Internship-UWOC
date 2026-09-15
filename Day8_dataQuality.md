# Week 2 (Day 2) — Data Quality & Class Audit

**Dataset:** `anaotherGUI/1-2NTU.csv` (8,512 rows: `labels`, `feature`, `bit`)

## 1. Missing Values, Duplicates, Invalid Values

- **Missing values:** None in any column.
- **Duplicate rows:** 8,486 out of 8,512 rows are exact duplicates — only 26 unique rows exist.
  - **Root cause:** The receiver's ADC (analog-to-digital converter) has limited resolution and can only output 26 distinct signal levels. Repeated identical readings are legitimate, repeated hardware measurements — not logging errors or data entry mistakes.
  - **Action:** Duplicates should **not** be removed. Doing so would shrink the dataset from 8,512 rows to 26, destroying class balance and any ability to train/evaluate a model meaningfully.
- **Invalid values:** `labels` and `bit` both only contain `{0, 1}` — no invalid entries found.

## 2. Class-Wise Summary Statistics

| Label | Count | Mean | Std Dev | Min | Max |
|---|---|---|---|---|---|
| 0 | 5,321 | 0.3485 | 0.0042 | 0.3369 | 0.3662 |
| 1 | 3,191 | 0.5293 | 0.0090 | 0.5176 | 0.6250 |

- Class balance: ~62.5% label 0, ~37.5% label 1 — moderately imbalanced but not extreme.
- Label 0 values are tightly clustered (very low std) between 0.337–0.366.
- Label 1 values are more spread out (0.518–0.625) but still completely non-overlapping with Label 0.

## 3. Leakage Risks & Condition-Specific Imbalance

- **Leakage risk identified:** The `bit` column agrees with the true `labels` column 100% of the time (perfect agreement). This means `bit` is a direct restatement of the answer (from an existing threshold-based classifier) and **must be excluded from any model's input features** — using it would let a model "see" the answer.
- **Condition-specific imbalance:** All 8,512 rows come from a single turbidity condition — **1–2 NTU**. No raw data exists for other turbidity ranges (2–3, 3–4, 4–5, 5–6 NTU); those exist in the source project only as result images, not usable data. This means any conclusions about robustness across turbidity levels cannot yet be validated with real data, and this dataset alone cannot demonstrate scenarios where ML would outperform a simple threshold, since the classes are perfectly separable at this condition.

## 4. Proposed Cleaning Rules

1. Do not remove duplicate rows — they reflect legitimate repeated ADC readings, not errors.
2. Exclude the `bit` column from model input features; it may only be used as a reference/comparison baseline, never as a predictor.
3. Retain all 26 unique signal levels; no outlier removal needed, as `feature` values fall in tight, well-defined ranges.
4. Before drawing any conclusions about robustness to turbidity, flag to the mentor that raw data for higher NTU levels is still missing, and treat single-condition results as preliminary/pipeline validation rather than final findings.
