# PRD — Credit Card Fraud Detection (Data Science P2)

## Project Overview

Validation, preprocessing, and modeling of the ULB Credit Card Fraud dataset.
Research reference: proposed sampling algorithm + deep ensemble learning.

Dataset: `creditcard_clean.csv` (283,726 rows x 31 cols, dupes removed)
Notebook: `notebook/creditcard_validation.ipynb`

---

## Current State

### Files

- `creditcard_clean.csv` — cleaned dataset (deduplicated from original 284,807 rows)
- `notebook/creditcard_validation.ipynb` — validation & verification notebook

### Notebook Contents (v1)

1. Context + relation to research paper
2. Column descriptions table (all 31 columns named)
3. 12 validation checks vs Kaggle spec
4. Duplicate row removal (1,081 removed)
5. Post-clean verification
6. Save outputs (semicolon-delimited, UTF-8 BOM)

### Validation Results

| Check            | Expected   | Result               |
| ---------------- | ---------- | -------------------- |
| Total rows       | 284,807    | PASS                 |
| Total columns    | 31         | PASS                 |
| Missing values   | 0          | PASS                 |
| Fraud count      | 492        | PASS                 |
| Legitimate count | 284,315    | PASS                 |
| Fraud ratio      | ~0.172%    | PASS                 |
| Time min         | 0          | PASS                 |
| Time max         | <=172,800s | PASS                 |
| Negative Amount  | 0          | PASS                 |
| Class values     | only 0/1   | PASS                 |
| V1-V28 mean      | ~0.0       | PASS                 |
| Duplicate rows   | 0 (ideal)  | WARN — 1,081 removed |

---

## Requested Changes / New Features

> Add your requests below. One item per bullet. I will pick these up and implement them.

<!-- Example:
- [ ] Add EDA notebook with class distribution plot and Amount histogram
- [ ] Add SMOTE oversampling cell to the validation notebook
- [ ] Train a baseline Random Forest model
-->

-

---

# Stage 2 — Data Cleaning

> Status: **requirements only**. No implementation code in this stage.
> Every cleaning decision listed here must be executed and logged before any EDA result is treated as final.

## 2.1 Objective

Produce a single trustworthy, analysis-ready dataset derived from `creditcard_clean.csv` by:

- eliminating structural defects (duplicates, wrong dtypes, malformed values),
- documenting every removal, replacement, or transformation with an explicit reason,
- preserving all signal relevant to fraud detection, especially the minority class.

**Guiding rule:** unusual is not the same as wrong. No row, column, or value may be dropped,
replaced, or transformed without a written justification recorded in the cleaning log
(`docs/cleaning_decisions.md`). The reason must be recorded _before_ the operation, not reconstructed after.

Deliverable of this stage: `data/creditcard_analysis_ready.csv` + `docs/cleaning_decisions.md`.

## 2.2 Verified Baseline (source: `creditcard_clean.csv`, read with `sep=';'`, `encoding='utf-8-sig'`)

These values are measured from the current file and are the reference point for every before/after comparison.

| Property                                  | Verified value                                                                     |
| ----------------------------------------- | ---------------------------------------------------------------------------------- | ---- | ---------------------- |
| Shape                                     | 283,726 rows × 31 columns                                                          |
| Columns                                   | `Time`, `V1`–`V28`, `Amount`, `Class`                                              |
| dtypes                                    | `Time`, `V1`–`V28`, `Amount` → float64; `Class` → int64                            |
| Missing values                            | 0 (all columns)                                                                    |
| Exact duplicate rows                      | 0                                                                                  |
| Rows duplicated on `V1`–`V28` (+`Amount`) | 8,063 rows / 4,383 groups (12,446 rows involved)                                   |
| `Class` = 0 (legitimate)                  | 283,253                                                                            |
| `Class` = 1 (fraud)                       | 473                                                                                |
| Fraud ratio                               | 0.16671%                                                                           |
| `Time` range                              | 0 → 172,792 s (≈47.998 h), integer-valued, 124,592 unique values                   |
| `Amount` range                            | 0 → 25,691.16; mean 88.47; median 22.00; std 250.40; Q1 5.60; Q3 77.51; skew 16.98 |
| `Amount` = 0 rows                         | 1,808 (1,783 legitimate, 25 fraud)                                                 |
| `Amount` < 0 rows                         | 0                                                                                  |
| `V1`–`V28` means                          | all                                                                                | mean | < 0.006 (PCA-centered) |
| `V1`–`V28` std                            | 1.948 (`V1`) down to 0.328 (`V28`) — **not** unit-scaled                           |
| Native categorical columns                | none; `Class` is the only discrete variable                                        |
| Original file                             | 284,807 rows → 1,081 exact duplicates already removed (19 fraud, 1,062 legitimate) |

## 2.3 Cleaning Steps

Each step below specifies: what is checked, why, how it is handled, the disposition
(remove / replace / transform / retain), and the impact on later analysis.

### C1 — Missing value detection and handling

- **Check:** per-column count and percentage of null / `NaN` / empty string / `"NA"`-like tokens across all 31 columns, after forcing `na_values` on read.
- **Why:** hidden nulls arriving as strings silently become object dtype and break numeric aggregation; a nonzero null count in `Amount` or `Class` would invalidate the current "missing values: PASS" claim.
- **Handle:** `isnull().sum()` per column, plus a scan for whitespace-only and sentinel strings (`?`, `-`, `null`, `NaN`, `n/a`) using a dtype-aware coercion probe.
- **Disposition:** verified baseline is 0 missing values → **retain all rows**. If any missing value appears in a re-sourced file: `Amount`/`Time` missing → investigate source (imputation on a numeric PCA dataset is only acceptable if <1% and documented); `Class` missing → row must be **removed** with the count logged, because an unlabeled row cannot be used in a supervised target analysis.
- **Impact:** a null in `Class` would change the class balance; a null in `Amount` would distort mean/median/skew and the fraud-vs-amount comparison.

### C2 — Duplicate detection and handling

- **Check:** three levels of duplication:
  1. exact full-row duplicates (`df.duplicated()`) — verified 0;
  2. duplicates on `V1`–`V28` only — verified 8,063 rows in 4,383 groups;
  3. duplicates on `V1`–`V28` + `Amount` — verified 8,063 rows (identical to level 2, i.e. matched groups also share `Amount`).
- **Why:** exact duplicates inflate sample weight and leak across a train/test split. The `V`-level duplicates need separate scrutiny: they are _not_ exact duplicates — within every one of the 4,383 groups the `Time` value differs (observed group time-spans of 2 s up to 77 s), while `Amount`, `Class`, and all 28 PCA components are identical. Two readings are possible: (a) the same underlying transaction recorded more than once with a differing timestamp, or (b) genuinely distinct transactions whose PCA projection coincides. Collisions in 28 PCA dimensions are implausible, so (a) is the working hypothesis — but it must be confirmed, not assumed.
- **Handle:** quantify both levels; inspect the group structure (`Time.nunique()`, `Class.nunique()`, `Amount.nunique()` per group — verified: 1 unique `Class` and 1 unique `Amount` per group); decide per level.
- **Disposition:** exact duplicates → **remove** (already done). `V`-level duplicate groups → **retain pending analysis**: do not drop yet. Because all rows in a group carry identical `Class` and `Amount`, removing the extra rows would not rebalance classes or bias `Amount`, but it _would_ delete legitimate-looking volume; and the group members carry different `Time` values, which is the only differing signal. Log the group count, the removed-if-any count, and the reasoning. If resolution is adopted, record whether the kept row is the first, last, or median-`Time` member.
- **Impact:** deduplicating the group level would reduce the dataset by up to 8,063 rows (~2.8%) and would need a re-stated fraud ratio; leaving them in risks the same transaction appearing in both train and test, which inflates reported performance at the modeling stage. This trade-off must be resolved and written down, not silently chosen.

### C3 — Data type validation and conversion

- **Check:** declared dtype vs. semantic type for every column. `Time` is float64 although verified integer-valued (all values `% 1 == 0`); `Class` is int64 with exactly 2 unique values; `V1`–`V28` and `Amount` are float64.
- **Why:** `Time` as float invites fractional-second values that do not exist in the source and makes time-bucket operations ambiguous; `Class` must be discrete for any grouped or binary comparison.
- **Handle:** verify integrality of `Time` across all rows, then convert. Cast `Class` explicitly.
- **Disposition:** **transform** `Time` → integer seconds (documented as a type correction, not a value change, only if the integrality check passes on every row). **Transform** `Class` → int8 (or keep int64 if downstream tooling prefers; the cast is cosmetic and must be listed). All PCA components and `Amount` **retain** float64 — do not downcast, the precision is used.
- **Impact:** type corrections prevent silent aggregation bugs in the time-based EDA section (hour-of-day, day split). No analytical result changes from these casts.

### C4 — Inconsistent value and formatting detection

- **Check:** for every column, scan for values that are non-numeric strings, mixed separators (decimal comma vs point), thousands separators, stray whitespace, and case inconsistency in any text-like field.
- **Why:** the file is locale-formatted for Excel (`;` delimiter, UTF-8 BOM). A locale-mismatched re-export can turn `1,234.56` into a string or shift `Amount` by a factor of 1000.
- **Handle:** attempt numeric coercion on all 31 columns and compare the coerced null count against the original; flag any column where coercion introduces nulls.
- **Disposition:** verified baseline shows every value already parses as numeric → **retain**. If a re-source introduces coercion failures, the affected rows are **flagged first**, quotations resolved, and only then **replaced** with the corrected numeric value — never silently dropped.
- **Impact:** a mis-parsed `Amount` column would invalidate every amount-based statistic and the fraud-vs-legitimate amount comparison; this check is the guard against that.

### C5 — Categorical value normalization

- **Check:** the dataset has no native categorical string columns. The only discrete variables are `Class` (verified: exactly {0, 1}) and `Time` (a continuous counter, not a category).
- **Why:** the request for categorical normalization applies to derived categoricals created during EDA (time-of-day bucket, amount band), not to raw columns; normalizing at the raw stage would be inventing structure.
- **Handle:** confirm `Class` domain is exactly {0, 1} with no other value; confirm no column is object/string dtype.
- **Disposition:** **retain** all columns as-is. `Class` is **transformed** only into a boolean/int8 representation if needed and the mapping `1 = fraud, 0 = legitimate` is preserved exactly. Any _derived_ categorical (day part, amount band) must be created in the EDA stage with its bin edges and label order documented, never here.
- **Impact:** guaranteeing the label domain prevents a mislabeled class from silently entering the target analysis; no data changes.

### C6 — Numerical data validation

- **Check:** for `Amount` and each of `V1`–`V28`: min, max, mean, std, presence of negative values, presence of exact zeros, and count of infinite values.
- **Why:** `Amount` is a monetary value with a hard lower bound of 0 (verified min 0.0, no negatives). The `V` columns are PCA outputs and legitimately span negative values, so a negative `V` value is _not_ an error — this distinction must be explicit so no cleaning step "fixes" PCA components.
- **Handle:** `Amount`: assert `min >= 0`, assert no `inf`. `V1`–`V28`: record observed ranges (e.g. `V1` −56.41 → 2.45, `V5` −113.74 → 34.80, `V7` −43.56 → 120.59) as _expected_ PCA spread, assert no `inf`/`NaN`.
- **Disposition:** **retain** everything. Verified: `Amount` min 0 with 1,808 exact-zero rows and 0 negative rows; no infinities expected — confirm and log. Zero-amount transactions are **retained** (see C10).
- **Impact:** establishes the numeric floor/ceiling so that the outlier step (C7) can distinguish _statistical_ extremes from _invalid_ values.

### C7 — Outlier detection and handling

- **Check:** IQR rule (Q1 − 1.5·IQR, Q3 + 1.5·IQR) on every numeric column, plus a robust z-score / MAD cross-check on `Amount`. Report outlier counts per column, not a single global number.
- **Why:** measured IQR outlier counts are high and uneven — `Amount` 31,685 rows (11.2%), `V27` 38,799, `V28` 30,094, `V20` 27,553, `V8` 23,904, `V6` 22,886, down to `V11` 735 and `Time` 0. High rates on PCA components are _expected_ for a centered, non-normal projection; treating them as errors would delete real fraud signal.
- **Handle:** compute per-column IQR bounds and outlier counts; for `Class == 1` rows, compare outlier rates against `Class == 0` rows to test whether "outlier" is informative rather than defective. Also check `Amount` with a log scale and MAD z-score to confirm the same rows flag under both methods.
- **Disposition:** **retain all outliers.** No winsorization, no trimming, no removal at this stage. Rationale: `Amount` is right-skewed (skew 16.98, median 22.00 vs mean 88.47) and its tail contains the highest-value fraud cases (fraud `Amount` reaches 2,125.87 with median only 9.82). Outlier treatment belongs to the modeling stage as an explicit, tested decision — not to cleaning. If any outlier is later removed, the record must state the column, the bound, the row count, and the business reason.
- **Impact:** retaining outliers preserves the fraud signal; documenting the rates prevents a later reader from assuming the data was cleaned of extremes.

### C8 — Irrelevant or redundant column identification

- **Check:** per column: variance, fraction of unique values, correlation with other columns, and domain usefulness. Verified unique-value counts: every `V` column has 275,663 unique values (identical count across all 28 — consistent with PCA of a dense continuous space); `Time` 124,592; `Amount` 32,767; `Class` 2.
- **Why:** a constant or near-constant column carries no information; a pair of near-duplicate columns carries redundant information. Neither has been demonstrated in this dataset yet.
- **Handle:** assert no zero-variance column (verified: no `V` column has std 0); compute the full 30×30 correlation matrix among `Time`, `V1`–`V28`, `Amount` and flag any |r| > 0.9 pair; report each flagged pair with its coefficient.
- **Disposition:** **retain all 31 columns.** No column is currently provably irrelevant. `Time` is the only candidate for exclusion — it is a relative counter with no intrinsic meaning once the two-day window is bucketed — but it must **not** be removed before the EDA stage, because the fraud-vs-time relationship is a primary EDA question. Any eventual drop requires the correlation/importance evidence recorded first.
- **Impact:** an unjustified drop of `Time` or of a low-variance `V` component could remove the only weak signal separating the 473 fraud rows from 283,253 legitimate rows. The rule here is: no column leaves the dataset during cleaning.

### C9 — Invalid or impossible value detection

- **Check:** logical constraint assertions across the dataset:
  1. `Class` ∈ {0, 1} — verified, no other value;
  2. `Amount` ≥ 0 — verified min 0.0, 0 negative rows;
  3. `Time` ∈ [0, 172,792] and monotone-consistent with a 48-hour window — verified max 172,792 s ≈ 47.998 h;
  4. no `inf` / `-inf` / `NaN` anywhere;
  5. no `Time` value below 0 or implying the recording window was exceeded.
- **Why:** these are the only hard domain constraints the dataset supports. Amount can be zero (an authorized hold can post 0) but cannot be negative; the recording window is stated as two consecutive days in the source documentation, so a value above 172,800 s would indicate a corrupt row.
- **Handle:** assert each constraint in code and report a count of violating rows per constraint.
- **Disposition:** verified zero violations → **retain all rows**. If a constraint is violated on a re-sourced file: negative `Amount` or out-of-window `Time` → **remove the row** with the count and the offending values logged. `Class` outside {0,1} → **flag and halt**, because the correct label cannot be recovered by inference.
- **Impact:** constraint violations in `Amount` or `Time` would corrupt the amount distribution and the time-based split; a bad `Class` value would poison the target analysis. Assert-early prevents all three.

### C10 — Dataset-specific cleaning steps

These follow from the properties measured above and have no generic analogue.

- **C10.1 — Zero-amount transactions.** 1,808 rows have `Amount == 0` (1,783 legitimate, 25 fraud). **Retain.** Justification: 25 of them are labeled fraud, i.e. a zero-amount transaction is a real fraud pattern in this data, not a missing-value placeholder. Removing them would delete confirmed positives. The zero-amount count must be reported in the data quality summary so that any later amount-based model documents how it treats them.
- **C10.2 — PCA components are not standardized.** `V1`–`V28` have unit-centered means but non-unit std (1.948 down to 0.328). **Retain as-is; document.** Do not rescale during cleaning — rescaling is a modeling-stage decision that changes the meaning of the components and must be justified there. The non-unit std is recorded now so that no later reader assumes the components are pre-scaled.
- **C10.3 — `Amount` skew.** skew 16.98 with median 22.00 vs mean 88.47. **Retain raw; document.** Do not apply a log transform during cleaning. The transformation, if used, must be applied at modeling time and evaluated, because the fraud subpopulation has a very different amount distribution (fraud median 9.82, mean 123.87, max 2,125.87) and a naive log transform changes the shape of both classes differently.
- **C10.4 — Deduplication already applied is not reversible in the current file.** The original 284,807 rows contained 1,081 exact duplicates; the current file has 283,726. Consequence: the delivered fraud count is 473, not the documented Kaggle figure of 492 — 19 fraud rows were removed as duplicates. **Retain the current file**; document the 19-row fraud reduction explicitly in the data quality summary, and note that any figure quoted from the literature (492 frauds, 284,807 rows, 0.172%) refers to the _pre-deduplication_ file and is not directly comparable. This discrepancy must be visible to whoever reads the EDA results.
- **C10.5 — Grouped `V`-level duplicates are unresolved.** Per C2, 4,383 groups of rows share identical `V1`–`V28`, identical `Amount`, and identical `Class`, differing only in `Time`. **Retain pending a documented decision.** The decision and its rationale must be written into `docs/cleaning_decisions.md` before the modeling stage, because it determines whether a time-based or a group-aware train/test split is required.
- **C10.6 — Class imbalance is extreme and must not be corrected in cleaning.** 473 fraud vs 283,253 legitimate = 0.16671%. **Retain the imbalance.** No resampling, no SMOTE, no undersampling, no class weighting during cleaning — those are modeling-stage interventions that need their own validation. The cleaned dataset must preserve the natural class ratio so that the imbalance itself is measurable in EDA.

## 2.4 Required Cleaning Log

`docs/cleaning_decisions.md` must contain one entry per executed operation with these fields:

| Field                 | Content                                                                               |
| --------------------- | ------------------------------------------------------------------------------------- |
| Step ID               | C1–C10.x                                                                              |
| Operation             | what was done                                                                         |
| Rows/columns affected | exact counts, before → after                                                          |
| Rationale             | why this operation was necessary, in terms of the verified property that motivated it |
| Evidence              | the command/query result that triggered the decision                                  |
| Reversibility         | whether the original data can be recovered                                            |
| Impact on analysis    | which downstream statistic is affected, and how                                       |

---

# Stage 3 — Exploratory Data Analysis (EDA)

> Status: **requirements only**. No implementation code in this stage.
> EDA runs on the output of Stage 2 (`data/creditcard_analysis_ready.csv`).
> Data Cleaning and EDA are separate stages with separate deliverables; EDA consumes Cleaning's output and must not silently change it. If EDA surfaces a data defect, it goes back to Stage 2 as a new numbered step.

## 3.1 Objective

Characterize the structure, distribution, and relationships of the cleaned dataset; state every
finding as a verified observation with its supporting number; and produce an explicit shortlist of
features and issues to carry into the modeling stage.

## 3.2 Dataset Overview

- **Dimensions:** confirm the shape of the Stage 2 output and report the delta against 283,726 × 31. Any row loss must be traceable to a Stage 2 step ID.
- **Column descriptions:** reproduce the 31-column table (`Time` = seconds since first transaction; `V1`–`V28` = anonymous PCA components; `Amount` = transaction value in EUR; `Class` = fraud label, 1 = fraud, 0 = legitimate). State that the original features behind `V1`–`V28` are confidential and therefore not interpretable — every finding about a `V` column must be phrased as "PCA component N", never as a business concept.
- **Data types:** report post-cleaning dtypes and confirm they match the Stage 2 decisions (`Time` integer seconds, `Class` discrete, `Amount` and `V1`–`V28` float).
- **Numerical vs categorical:** state explicitly that there are 30 numerical columns and exactly 1 discrete column (`Class`); that there are no native categorical columns; and that any categorical variable used below is derived, with its derivation documented.
- **Summary statistics:** `describe()` for all numeric columns — count, mean, std, min, Q1, median, Q3, max — plus skew and kurtosis per column. Report `Amount` separately at full precision (mean 88.47, median 22.00, std 250.40, skew 16.98) because its scale differs from the PCA components.

## 3.3 Univariate Analysis

### Numerical variables (`Time`, `V1`–`V28`, `Amount`)

For each of the 30 columns report: mean, median, standard deviation, min, max, Q1, Q3, skew, kurtosis, and histogram.

- **Distribution:** plot a histogram for every column. For `Amount` and `Time`, plot both the raw distribution and a log-scaled version, because `Amount` is strongly right-skewed (16.98) and `Time` is uniform across 0–172,792 s. Note the visible shape: `Amount` concentrated at low values with a long sparse tail; the PCA components roughly bell-shaped with different spreads.
- **Boxplot:** for every column, and specifically for `Amount` — the boxplot is the primary view of the 31,685 IQR-flagged `Amount` rows.
- **Interpretation required** after each block, not after each individual chart.

### Categorical / discrete variables (`Class`)

- **Frequency counts:** 0 → 283,253; 1 → 473 (pre-cleaning verified values; restate post-cleaning).
- **Percentage distribution:** 99.83329% legitimate, 0.16671% fraud.
- **Bar chart:** two-bar count plot, plus a log-scale companion chart so the 473-row fraud bar is visible next to the 283,253-row legitimate bar. A single linear-scale bar chart renders the fraud class as an invisible sliver — the log companion is required, not optional.
- No other discrete variable exists. If a derived categorical is used (e.g. time-of-day bucket, amount band), its bin edges, label order, and the reason for the chosen edges must be stated on the chart and in the text.

## 3.4 Bivariate Analysis

### Numerical vs numerical

- **Scatter plots:** `Time` vs `Amount`; selected `V` vs `V` for components that the correlation step (3.5) identifies as related; `Amount` vs each `V` component where a relationship is hypothesized. With 283k points, use sampling plus alpha transparency or hexbin — an unreadable dense scatter does not count as a visualization.
- **Correlation analysis:** Pearson correlation between all numeric pairs; report the top absolute correlations as a ranked table, not a bare matrix. If any pair exceeds |r| > 0.9, list it in the cleaning log as a redundancy candidate (C8) rather than dropping it in EDA.
- **Class-conditional correlation:** compare correlations computed on `Class == 1` rows against `Class == 0` rows. A relationship that exists only inside the fraud subpopulation is a candidate signal and must be called out separately.

### Categorical vs numerical

`Class` (binary) vs each of `Time`, `Amount`, `V1`–`V28`:

- **Boxplots:** one boxplot per numeric column, split by `Class`. For `Amount`, the class-conditional comparison is a required headline result: verified pre-cleaning medians are 22.00 (legitimate) vs 9.82 (fraud), and means 88.41 vs 123.87 — the mean and median disagree in direction, so the finding must be stated as "fraud transactions have a _lower_ median but a _higher_ mean amount, indicating a heavier tail", not as "fraud has higher amounts".
- **Group statistics table:** for every numeric column, per-class count, mean, median, std, min, max, and the difference between classes. This table is a required deliverable.
- **Bar charts:** mean/median `Amount` by `Class`; per-hour transaction count by `Class` after deriving an hour-of-day variable from `Time`.
- **Time-bucketed counts:** `Time` spans 47.998 h. Bucket into hours and plot counts by class, to test whether fraud is concentrated in a specific part of the two-day window.

### Categorical vs categorical

- Only one discrete variable exists (`Class`), so a raw contingency table is not possible. **State this explicitly** rather than fabricating a pair. If a derived categorical is created (time-of-day bucket, amount band), cross it with `Class` in a contingency table with row percentages and a bar/mosaic chart.

## 3.5 Multivariate Analysis

- **Correlation matrix:** 30×30 over `Time`, `V1`–`V28`, `Amount`.
- **Heatmap:** annotated or colorbar-labelled, with a stated scale and sign convention. If the matrix is too dense to read at 30×30, split it into blocks (e.g. `V1`–`V14`, `V15`–`V28`, plus `Time`/`Amount`) and say so.
- **PCA context:** the components are already orthogonal by construction, so near-zero correlations are expected and are _not_ a finding. State this so that the heatmap is not misread. Any non-trivial correlation between two components indicates something about the preprocessing, not a business relationship.
- **Class-conditional multivariate views:** correlation heatmap for `Class == 1` against `Class == 0`; a 2D projection (e.g. selected pair of components, or a new PCA for visualization only) with points colored by `Class` and sampled to remain readable.
- **Pattern identification:** report which components show the largest class-conditional separation, with the numeric evidence (e.g. mean shift, overlap width, or a per-class distribution comparison per component). Rank them; do not assert separation without a number.

## 3.6 Target Variable Analysis

The dataset **does contain a target variable**: `Class` (binary classification — fraud detection).

- **Target distribution:** 473 fraud vs 283,253 legitimate in the pre-cleaning verified file; restate on the cleaned file. Report absolute counts and percentages.
- **Class balance / imbalance:** severe. Fraud is 0.16671% of rows, i.e. a 1:599 ratio, which is roughly 599 legitimate rows per fraud row. State this ratio explicitly. Report that no correction was applied in cleaning (C10.6) and that the imbalance is a modeling-stage problem, not a cleaning-stage problem.
- **Distribution and outliers per class:** for every numeric column, report per-class min/Q1/median/Q3/max, and the count of IQR outliers inside each class. For `Amount`: legitimate median 22.00, fraud median 9.82; legitimate max 25,691.16, fraud max 2,125.87 — i.e. the largest transactions are legitimate, while fraud sits lower in the middle but with a heavier upper tail relative to its own median.
- **Zero-amount rows by class:** 1,783 legitimate vs 25 fraud. Report how the target behaves inside this subset.
- **Target vs important features:** for the components that show the strongest class separation in 3.5, plot class-conditional distributions and report the numeric separation. Rank the features by separation strength.
- **Class imbalance implications to state:** a model that predicts "legitimate" for every row scores 99.833% accuracy while catching 0 fraud. Therefore accuracy is a misleading metric for this target, and the EDA must say so in writing so the modeling stage is not evaluated by accuracy alone.
- **Target leakage to check:** explicitly examine three risks and report a verdict on each:
  1. **Duplicate leakage** — rows sharing identical `V1`–`V28` + `Amount` (8,063 rows / 4,383 groups, per C2/C10.5) would leak across a naive random split. Report the affected row count and state that it constrains the valid split strategy.
  2. **Temporal leakage** — `Time` is a continuous 48-hour record; a random split mixes earlier and later transactions and lets a model see the future. Report whether the fraud rate varies across the time window, which determines whether a temporal split is needed.
  3. **`Class`-derived features** — confirm no column is derived from the label (all non-label columns are PCA components of anonymized original features plus `Time` and `Amount`, so this is expected to be clean; verify and record).
- **No unsupported claims:** every statement in this section must cite its number from the analysis output. No hypothesized fraud pattern may be written into the PRD as a finding.

## 3.7 Visualization Requirements

Every chart produced in this stage must satisfy all of the following:

- **Title:** descriptive, stating the variable(s) and the comparison. "Amount distribution by class", not "chart1".
- **Axis labels:** both axes labelled with the variable name and its unit — `Amount` in EUR, `Time` in seconds (and in hours when bucketed), `V` columns as "PCA component N (unitless)".
- **Chart type:** matched to the data — histogram for distributions, boxplot for spread and outliers, bar chart for counts and class comparisons, scatter/hexbin for numeric pairs, heatmap for the correlation matrix. Never use a pie chart for the 99.83/0.17 class split.
- **Readability:** legible font sizes, no label overlap, log scale used where the raw scale hides the minority class, sampled or alpha-blended scatter for 283k-point plots.
- **Units** stated wherever a quantity has one (`Amount` in EUR; `Time` in seconds or hours).
- **No unnecessary complexity:** no 3D charts, no dual-axis charts unless a genuine dual-unit comparison is required, no decorative gradients, no more than one message per figure.
- **Interpretation after important visualizations:** a written statement of what the chart shows, in plain terms, with the supporting numbers. A chart without an interpretation is not a deliverable.

**Purpose test:** each visualization must answer a specific question from 3.8. State which question, in the chart caption or the surrounding text. Charts that answer no listed question are removed rather than shipped.

## 3.8 EDA Questions to Answer

These are adapted to this dataset's measured properties, not generic prompts.

**Structure**

1. What is the exact shape and dtype profile of the cleaned dataset, and does it match the Stage 2 baseline of 283,726 × 31?
2. Which columns are continuous (`Time`, `V1`–`V28`, `Amount`) and which are discrete (`Class` only)? Are there any derived categoricals, and how were their bins chosen?

**Data quality** 3. Are there any missing values or inf values in the cleaned file, and does the count still read 0? 4. Do the 4,383 grouped `V`-level duplicate rows (differing only in `Time`) materially change any distribution when excluded? Quantify the answer instead of assuming it. 5. How many rows sit outside the IQR bounds per column, and are those rows concentrated in `Class == 1` or `Class == 0`? (Expected scale: `Amount` ≈ 31,685 rows, `V27` ≈ 38,799, `V11` ≈ 735, `Time` = 0 — verify on the cleaned file.)

**Univariate** 6. Which columns have the highest variability? Report by standard deviation and by coefficient of variation, noting that `Amount` (std 250.40) and `Time` (std 47,481 s) are on fundamentally different scales from the PCA components. 7. What is the shape of the `Amount` distribution, and what does skew 16.98 with median 22.00 vs mean 88.47 imply about how many large transactions exist? 8. Is `Time` uniformly spread across the 47.998-hour window, or are there gaps and bursts? 9. Do the 28 PCA components all look roughly Gaussian, or do some deviate? Which ones, and how?

**Bivariate** 10. Is there a relationship between `Time` and `Amount`? Does transaction size vary across the two-day window? 11. Do fraud transactions cluster at particular times within the window? 12. How does `Amount` differ between classes when compared by median against by mean, and what does the discrepancy (fraud median 9.82 vs mean 123.87) reveal about the fraud amount tail? 13. Do the 1,808 zero-amount rows (`Amount == 0`) behave differently from positive-amount rows within each class?

**Multivariate** 14. Which pairs of columns have a non-trivial correlation, given that the PCA components are orthogonal by construction? Any such pair is a preprocessing artifact and must be flagged, not interpreted as a business relationship. 15. Which components separate the two classes most strongly, ranked with numeric evidence? 16. Does the correlation structure differ between the fraud and legitimate subpopulations?

**Target** 17. What is the exact class ratio in the cleaned file, and how far is it from 1:1? (Baseline: 473 : 283,253, ratio ≈ 1:599, fraud 0.16671%.) 18. Is accuracy a viable metric for this target? (No — state the 99.833% majority-class baseline explicitly.) 19. Is there duplicate-based or temporal target leakage risk? Report the row counts for each risk (8,063 rows in 4,383 groups for duplicate risk; the full 283k rows for temporal risk) and the mitigation each requires. 20. Which features appear potentially important for further analysis or modeling, ranked by evidence, and which are candidates for exclusion — with the evidence for each?

## 3.9 Expected Outputs

1. **Cleaned dataset** — `data/creditcard_analysis_ready.csv`, semicolon-delimited, UTF-8 BOM, with a row count traceable to Stage 2.
2. **Data quality summary** — one table per issue type (missing, duplicate, invalid, outlier, dtype) with counts, percentages, disposition, and the Stage 2 step ID that governs it.
3. **Before-and-after cleaning statistics** — at minimum: rows before/after; fraud and legitimate counts before/after; fraud ratio before/after; column count before/after; duplicate count at each level; and the explicit note that the delivered file has 473 fraud rows against the literature's 492 because 19 fraud duplicates were removed.
4. **Descriptive statistics** — full summary table for all 30 numeric columns (count, mean, std, min, Q1, median, Q3, max, skew, kurtosis) plus the per-class group statistics table.
5. **EDA visualizations** — the full set defined in 3.3–3.5, each with title, labelled axes, units, and a written interpretation, each mapped to a question in 3.8.
6. **Correlation analysis** — the 30×30 matrix, the annotated heatmap, the ranked top-correlation table, and the class-conditional comparison.
7. **Key findings and observations** — a numbered list where every finding cites its supporting number. No finding without a number.
8. **List of potential features for subsequent modeling** — every column, with its evidence-based ranking and its status (keep for modeling / candidate for exclusion / requires transformation at modeling time), plus the unresolved items (`Time` handling, the grouped duplicates, `Amount` skew) flagged as open decisions rather than resolved ones.
9. **Documentation of important decisions** — `docs/cleaning_decisions.md` (Stage 2 log, per 2.4) plus an EDA decisions section recording bin choices, sampling choices, exclusion choices, and any question in 3.8 that the available data could not answer.

## 3.10 Scope Boundary

This stage ends at characterization. It explicitly does **not** include and must not add:
model training, model evaluation, hyperparameter tuning, resampling or class-imbalance correction, feature scaling for models, train/test split construction, or deployment. Those belong to later stages and are separate PRD sections.

Findings that emerge here are **observations**, not conclusions about model performance.

---

## Conventions

- CSV exports: semicolon delimiter (`sep=';'`), UTF-8 BOM (`encoding='utf-8-sig'`)
- Notebooks go in `notebook/`
- CSV data files stay in project root (`P2/`)
- Notebook reads data via relative path `../creditcard_clean.csv`
  github repository: https://github.com/Khairil-12/Assignment_Creadit-card-fraud-detection.git
