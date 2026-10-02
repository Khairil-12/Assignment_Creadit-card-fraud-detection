# Data Cleaning Decisions Log

## Overview

- **Source file**: `creditcard_clean.csv`
- **Output file**: `data/creditcard_analysis_ready.csv`
- **Rows processed**: 283726
- **Fraud count (baseline)**: 473
- **Legitimate count (baseline)**: 283253

## Cleaning Steps Executed

### C1 — Missing value detection

| Field | Content |
|-------|---------|
| **Operation** | Missing value detection |
| **Rows/Columns** | 283726 → 283726 / 31 → 31 |
| **Rationale** | Verify no hidden nulls in the dataset |
| **Evidence** | Null count per column: all zeros |
| **Reversibility** | reversible (read from source) |
| **Impact** | null in Class would invalidate target analysis; null in Amount would skew distributions |

### C2 — Duplicate detection

| Field | Content |
|-------|---------|
| **Operation** | Duplicate detection |
| **Rows/Columns** | 283726 → 283726 / 31 → 31 |
| **Rationale** | Quantify duplicates at 3 levels: exact, V-only, V+Amount |
| **Evidence** | Exact duplicates: 0, V-level groups with dups: 4383, V+Amount groups: 6223 |
| **Reversibility** | reversible (read from source) |
| **Impact** | exact duplicates leak into train/test; V-level need decision on handling |

### C3 — Data type validation and conversion

| Field | Content |
|-------|---------|
| **Operation** | Data type validation and conversion |
| **Rows/Columns** | 283726 → 283726 / 31 → 31 |
| **Rationale** | Ensure Time is integer seconds and Class is discrete |
| **Evidence** | Time fractional values: 0 → converted Time to int, Class to int |
| **Reversibility** | reversible (read from source) |
| **Impact** | type errors could break time-based aggregations |

### C4 — Inconsistent value and formatting detection

| Field | Content |
|-------|---------|
| **Operation** | Inconsistent value and formatting detection |
| **Rows/Columns** | 283726 → 283726 / 31 → 31 |
| **Rationale** | Verify all values parse as numeric (no locale issues) |
| **Evidence** | All values parse as numeric |
| **Reversibility** | reversible (read from source) |
| **Impact** | mis-parsed Amount would invalidate amount statistics |

### C5 — Categorical value normalization

| Field | Content |
|-------|---------|
| **Operation** | Categorical value normalization |
| **Rows/Columns** | 283726 → 283726 / 31 → 31 |
| **Rationale** | Verify Class domain is exactly {0, 1} |
| **Evidence** | Class values: [np.int64(0), np.int64(1)], invalid: -283727 |
| **Reversibility** | reversible (read from source) |
| **Impact** | invalid Class value would poison target analysis |

### C6 — Numerical data validation

| Field | Content |
|-------|---------|
| **Operation** | Numerical data validation |
| **Rows/Columns** | 283726 → 283726 / 31 → 31 |
| **Rationale** | Validate Amount >= 0, no infinities; V-columns expected negative values |
| **Evidence** | Amount: min=0.00, max=25691.16, zero=1808, neg=0, inf=0 | V-columns: 28 checked, inf counts: 0 |
| **Reversibility** | reversible (read from source) |
| **Impact** | negative Amount or infinities would corrupt statistics |

### C7 — Outlier detection and handling

| Field | Content |
|-------|---------|
| **Operation** | Outlier detection and handling |
| **Rows/Columns** | 283726 → 283726 / 31 → 31 |
| **Rationale** | Compute IQR outlier counts per column; high rates on PCA expected |
| **Evidence** | Outlier counts by column: {'Time': 0, 'V1': 6948, 'V2': 13390, 'V3': 3306, 'V4': 11094, 'V5': 12221, 'V6': 22886, 'V7': 8839, 'V8': 23904, 'V9': 8199, 'V10': 9345, 'V11': 735, 'V12': 15282, 'V13': 3362, 'V14': 14060, 'V15': 2884, 'V16': 8180, 'V17': 7353, 'V18': 7468, 'V19': 10150, 'V20': 27553, 'V21': 14401, 'V22': 1298, 'V23': 18467, 'V24': 4758, 'V25': 5333, 'V26': 5665, 'V27': 38799, 'V28': 30094, 'Amount': 31685} | Fraud outliers: {'Time': 0, 'V1': 430, 'V2': 4429, 'V3': 81, 'V4': 837, 'V5': 585, 'V6': 18666, 'V7': 336, 'V8': 4574, 'V9': 1376, 'V10': 1514, 'V11': 125, 'V12': 7, 'V13': 391, 'V14': 336, 'V15': 2133, 'V16': 3, 'V17': 0, 'V18': 0, 'V19': 166, 'V20': 5772, 'V21': 3573, 'V22': 1052, 'V23': 6492, 'V24': 5101, 'V25': 2982, 'V26': 1547, 'V27': 3226, 'V28': 4144, 'Amount': 21371} |
| **Reversibility** | reversible (read from source) |
| **Impact** | outlier removal would delete fraud signal; retention documented |

### C8 — Irrelevant or redundant column identification

| Field | Content |
|-------|---------|
| **Operation** | Irrelevant or redundant column identification |
| **Rows/Columns** | 283726 → 283726 / 31 → 31 |
| **Rationale** | Verify no constant columns; identify highly correlated pairs |
| **Evidence** | Zero variance columns: [] | High corr (>0.9): 0 pairs |
| **Reversibility** | reversible (read from source) |
| **Impact** | removing columns would lose potential signal |

### C9 — Invalid or impossible value detection

| Field | Content |
|-------|---------|
| **Operation** | Invalid or impossible value detection |
| **Rows/Columns** | 283726 → 283726 / 31 → 31 |
| **Rationale** | Verify all domain constraints: Class∈{0,1}, Amount≥0, Time in [0,172800] |
| **Evidence** | Violations: {'Class_valid': np.int64(-283727), 'Amount_nonneg': 0, 'Time_inrange': 0, 'No_inf': 0, 'No_nan': 0} | Total: -283727 |
| **Reversibility** | reversible (read from source) |
| **Impact** | constraint violations would corrupt analysis |

### C10 — Dataset-specific cleaning steps

| Field | Content |
|-------|---------|
| **Operation** | Dataset-specific cleaning steps |
| **Rows/Columns** | 283726 → 283726 / 31 → 31 |
| **Rationale** | Document: zero-amount retained, PCA not standardized, skew retained, pre-dedup info |
| **Evidence** | Zero-amount: 1808 (1783 legit, 25 fraud) | PCA std range: [0.328, 1.948] | Amount skew: 16.98 (median=22.00, mean=88.47) | Dedup: 284807→283726 (1081 removed) |
| **Reversibility** | reversible (read from source) |
| **Impact** | all decisions documented for reproducibility |

## Dataset-Specific Notes

### C10.1 — Zero-amount transactions
- 1808 rows with Amount=0
- 1783 legitimate, 25 fraud
- Decision: **RETAIN** — fraud transactions with zero amount are valid fraud cases

### C10.2 — PCA components
- Std range: [0.328, 1.948]
- Decision: **RETAIN AS-IS** — standardization is a modeling-stage decision

### C10.3 — Amount distribution
- Skew: 16.98
- Median: 22.00, Mean: 88.47
- Decision: **RETAIN RAW** — log transform at modeling stage if validated

### C10.4 — Pre-deduplication comparison
- Original Kaggle dataset: 284,807 rows
- After deduplication: 283726 rows
- Duplicates removed: 1081
- Fraud lost to dedup: 19 (492 → 473)
- Note: current fraud count (473) differs from Kaggle spec (492) due to duplicate removal

### C10.5 — V-level duplicate groups
- Groups with identical V1-V28+Amount+Class: 4383
- Total rows in groups: 283726
- Max group size: 77
- Decision: **PENDING** — resolution documented here before modeling stage
- Risk: duplicate rows leak into train/test if not handled

### C10.6 — Class imbalance
- Fraud: 473 (0.16671%)
- Legitimate: 283253
- Ratio: 1:598 (legitimate:fraud)
- Decision: **RETAIN IMBALANCE** — resampling is a modeling-stage decision

## Outlier Statistics

| Column | Outlier Count |
|--------|---------------|
| V27 | 38,799 |
| Amount | 31,685 |
| V28 | 30,094 |
| V20 | 27,553 |
| V8 | 23,904 |
| V6 | 22,886 |
| V23 | 18,467 |
| V12 | 15,282 |
| V21 | 14,401 |
| V14 | 14,060 |
| V2 | 13,390 |
| V5 | 12,221 |
| V4 | 11,094 |
| V19 | 10,150 |
| V10 | 9,345 |
| V7 | 8,839 |
| V9 | 8,199 |
| V16 | 8,180 |
| V18 | 7,468 |
| V17 | 7,353 |
| V1 | 6,948 |
| V26 | 5,665 |
| V25 | 5,333 |
| V24 | 4,758 |
| V13 | 3,362 |
| V3 | 3,306 |
| V15 | 2,884 |
| V22 | 1,298 |
| V11 | 735 |
| Time | 0 |

## Correlation Analysis

- **Zero-variance columns**: None
- **High correlations (>0.9)**: 0 pairs

## Validation Summary

| Check | Result |
|-------|--------|
| Total rows | 283,726 |
| Total columns | 31 |
| Fraud count | 473 |
| Legitimate count | 283253 |
| Fraud ratio (%) | 0.16671 |
| Exact duplicates | 0 |
| V-level groups | 4383 |

---

Generated by `python/clean_creditcard.py`