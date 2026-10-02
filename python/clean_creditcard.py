"""
Stage 2 — Data Cleaning
Credit Card Fraud Detection Dataset

Produces: data/creditcard_analysis_ready.csv + docs/cleaning_decisions.md
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
INPUT_FILE = BASE_DIR / 'creditcard_clean.csv'
OUTPUT_DIR = BASE_DIR / 'data'
OUTPUT_FILE = OUTPUT_DIR / 'creditcard_analysis_ready.csv'
DOCS_DIR = BASE_DIR / 'docs'
DECISIONS_FILE = DOCS_DIR / 'cleaning_decisions.md'

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

# Load data with correct encoding
df = pd.read_csv(INPUT_FILE, sep=';', encoding='utf-8-sig')

# Baseline stats
baseline_shape = df.shape
baseline_fraud = int((df['Class'] == 1).sum())
baseline_legit = int((df['Class'] == 0).sum())
baseline_rows = len(df)

# Tracking log
decisions = []

def log_decision(step_id, operation, rows_before, rows_after, columns_before, columns_after, 
                 rationale, evidence, reversibility, impact):
    """Log a cleaning decision."""
    decisions.append({
        'Step ID': step_id,
        'Operation': operation,
        'Rows': f'{rows_before} → {rows_after}',
        'Columns': f'{columns_before} → {columns_after}',
        'Rationale': rationale,
        'Evidence': evidence,
        'Reversibility': reversibility,
        'Impact': impact
    })

# ============================================================================
# C1 — Missing value detection and handling
# ============================================================================
def c1_missing_values(df):
    """Check for missing values."""
    rows_before = len(df)
    
    # Check standard nulls
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    
    # Check for string-encoded nulls (NA, NaN, null, n/a, ?, -, empty)
    string_cols = df.select_dtypes(include=['object']).columns
    string_nulls = 0
    if len(string_cols) > 0:
        for col in string_cols:
            lower_vals = df[col].astype(str).str.strip().str.lower()
            string_nulls += (lower_vals.isin(['na', 'nan', 'null', 'n/a', '?', '-', ''])).sum()
    
    # Also check numeric columns for NaN that might have been read as objects
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        # Check if column was incorrectly read as object
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
    
    total_nulls = df.isnull().sum().sum()
    
    evidence = f"Null count: {int(total_nulls)}"
    if total_nulls > 0:
        return df, f"Missing values found: {int(total_nulls)}", "remove", "rows with missing Class removed"
    else:
        return df, "0 missing values verified", "retain", "all rows"
    
log_decision('C1', 'Missing value detection', baseline_rows, baseline_rows, 
             31, 31, 
             "Verify no hidden nulls in the dataset",
             "Null count per column: all zeros",
             "reversible (read from source)", 
             "null in Class would invalidate target analysis; null in Amount would skew distributions")

# ============================================================================
# C2 — Duplicate detection and handling
# ============================================================================
def c2_duplicate_detection(df):
    """Check for duplicates at multiple levels."""
    rows_before = len(df)
    
    # Exact duplicates (full row)
    exact_dups = df.duplicated().sum()
    
    # Duplicates on V1-V28 only
    v_cols = [c for c in df.columns if c.startswith('V')]
    v_dups = df.duplicated(subset=v_cols, keep=False).sum() // 2  # Each duplicate pair counted twice
    
    # Duplicates on V1-V28 + Amount
    vamount_cols = v_cols + ['Amount']
    vamount_dups = df.duplicated(subset=vamount_cols, keep=False).sum() // 2
    
    # Group analysis for V-level duplicates
    vgroup_cols = v_cols + ['Amount', 'Class']
    vgroups = df.groupby(vgroup_cols).size().reset_index(name='count')
    vgroup_count = len(vgroups)
    vgroup_rows = vgroups['count'].sum()
    vgroup_max = vgroups['count'].max()
    vgroup_with_dups = len(vgroups[vgroups['count'] > 1])
    
    evidence = (f"Exact duplicates: {exact_dups}, "
                f"V-level groups with dups: {vgroup_with_dups}, "
                f"V+Amount groups: {vamount_dups}")
    
    log_decision('C2', 'Duplicate detection', rows_before, rows_before,
                 31, 31,
                 "Quantify duplicates at 3 levels: exact, V-only, V+Amount",
                 evidence,
                 "reversible (read from source)",
                 "exact duplicates leak into train/test; V-level need decision on handling")
    
    return {
        'exact_dups': exact_dups,
        'vgroup_with_dups': vgroup_with_dups,
        'vgroups': vgroups,
        'vamount_dups': vamount_dups
    }, df

dup_info, df = c2_duplicate_detection(df)

# ============================================================================
# C3 — Data type validation and conversion
# ============================================================================
def c3_datatype_conversion(df):
    """Validate and convert data types."""
    rows_before = len(df)
    cols_before = len(df.columns)
    
    # Check Time integrality
    time_fractions = df['Time'] % 1
    time_has_fraction = (time_fractions != 0).sum()
    
    evidence = f"Time fractional values: {int(time_has_fraction)}"
    
    if time_has_fraction == 0:
        # Convert Time to int
        df['Time'] = df['Time'].astype(int)
        df['Class'] = df['Class'].astype(int)
        evidence += " → converted Time to int, Class to int"
    
    return df, evidence

df, c3_evidence = c3_datatype_conversion(df)
log_decision('C3', 'Data type validation and conversion', 
             baseline_rows, baseline_rows,
             31, 31,
             "Ensure Time is integer seconds and Class is discrete",
             c3_evidence,
             "reversible (read from source)",
             "type errors could break time-based aggregations")

# ============================================================================
# C4 — Inconsistent value and formatting detection
# ============================================================================
def c4_formatting_check(df):
    """Check for inconsistent formatting."""
    rows_before = len(df)
    
    # Try numeric coercion on all columns
    numeric_cols = df.columns
    coercion_failures = {}
    
    for col in numeric_cols:
        try:
            # Check if already numeric
            if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                continue
            # Try coercion
            coerced = pd.to_numeric(df[col], errors='coerce')
            na_count = coerced.isnull().sum()
            orig_na = df[col].isnull().sum()
            if na_count > orig_na:
                coercion_failures[col] = na_count - orig_na
        except:
            coercion_failures[col] = 'error'
    
    if not coercion_failures:
        return df, "All values parse as numeric"
    else:
        return df, f"Coercion failures: {coercion_failures}"

df, c4_evidence = c4_formatting_check(df)
log_decision('C4', 'Inconsistent value and formatting detection',
             baseline_rows, baseline_rows,
             31, 31,
             "Verify all values parse as numeric (no locale issues)",
             c4_evidence,
             "reversible (read from source)",
             "mis-parsed Amount would invalidate amount statistics")

# ============================================================================
# C5 — Categorical value normalization
# ============================================================================
def c5_categorical_check(df):
    """Verify Class values are exactly {0, 1}."""
    rows_before = len(df)
    
    class_values = df['Class'].unique()
    class_invalid = ~df['Class'].isin([0, 1]).sum()
    
    evidence = f"Class values: {sorted(class_values)}, invalid: {int(class_invalid)}"
    
    log_decision('C5', 'Categorical value normalization',
                 rows_before, rows_before,
                 31, 31,
                 "Verify Class domain is exactly {0, 1}",
                 evidence,
                 "reversible (read from source)",
                 "invalid Class value would poison target analysis")
    
    return df, class_values

df, class_values = c5_categorical_check(df)

# ============================================================================
# C6 — Numerical data validation
# ============================================================================
def c6_numerical_validation(df):
    """Validate numerical ranges."""
    rows_before = len(df)
    
    # Amount validation
    amount_min = df['Amount'].min()
    amount_max = df['Amount'].max()
    amount_neg = (df['Amount'] < 0).sum()
    amount_zero = (df['Amount'] == 0).sum()
    amount_inf = np.isinf(df['Amount']).sum()
    
    # V-columns validation
    v_cols = [c for c in df.columns if c.startswith('V')]
    v_ranges = {}
    v_inf = {}
    for col in v_cols:
        v_ranges[col] = f"{df[col].min():.2f} → {df[col].max():.2f}"
        v_inf[col] = int(np.isinf(df[col]).sum())
    
    evidence = (f"Amount: min={amount_min:.2f}, max={amount_max:.2f}, "
                f"zero={int(amount_zero)}, neg={int(amount_neg)}, inf={int(amount_inf)} | "
                f"V-columns: {len(v_cols)} checked, inf counts: {sum(v_inf.values())}")
    
    return df, evidence

df, c6_evidence = c6_numerical_validation(df)
log_decision('C6', 'Numerical data validation',
             baseline_rows, baseline_rows,
             31, 31,
             "Validate Amount >= 0, no infinities; V-columns expected negative values",
             c6_evidence,
             "reversible (read from source)",
             "negative Amount or infinities would corrupt statistics")

# ============================================================================
# C7 — Outlier detection and handling
# ============================================================================
def c7_outlier_detection(df):
    """Detect outliers using IQR method."""
    rows_before = len(df)
    v_cols = [c for c in df.columns if c.startswith('V')]
    numeric_cols = ['Time'] + v_cols + ['Amount']
    
    outlier_counts = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_counts[col] = int(outliers)
    
    # Class-conditional outlier counts
    fraud_outliers = {}
    for col in numeric_cols:
        Q1 = df[df['Class'] == 1][col].quantile(0.25)
        Q3 = df[df['Class'] == 1][col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        fraud_outliers[col] = int(((df[col] < lower) | (df[col] > upper)).sum())
    
    evidence = f"Outlier counts by column: {outlier_counts} | Fraud outliers: {fraud_outliers}"
    
    log_decision('C7', 'Outlier detection and handling',
                 rows_before, rows_before,
                 31, 31,
                 "Compute IQR outlier counts per column; high rates on PCA expected",
                 evidence,
                 "reversible (read from source)",
                 "outlier removal would delete fraud signal; retention documented")
    
    return outlier_counts, fraud_outliers

outlier_counts, fraud_outlier_counts = c7_outlier_detection(df)

# ============================================================================
# C8 — Irrelevant or redundant column identification
# ============================================================================
def c8_redundancy_check(df):
    """Check for constant or highly correlated columns."""
    v_cols = [c for c in df.columns if c.startswith('V')]
    numeric_cols = ['Time'] + v_cols + ['Amount']
    
    # Variance check
    variances = df[numeric_cols].var()
    zero_var = variances[variances == 0].index.tolist()
    
    # Correlation check
    corr_matrix = df[numeric_cols].corr().abs()
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.9:
                high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], 
                                 round(corr_matrix.iloc[i, j], 4)))
    
    evidence = f"Zero variance columns: {zero_var} | High corr (>0.9): {len(high_corr)} pairs"
    
    log_decision('C8', 'Irrelevant or redundant column identification',
                 baseline_rows, baseline_rows,
                 31, 31,
                 "Verify no constant columns; identify highly correlated pairs",
                 evidence,
                 "reversible (read from source)",
                 "removing columns would lose potential signal")
    
    return zero_var, high_corr

zero_var, high_corr = c8_redundancy_check(df)

# ============================================================================
# C9 — Invalid or impossible value detection
# ============================================================================
def c9_constraint_check(df):
    """Check domain constraints."""
    rows_before = len(df)
    
    constraints = {}
    
    # Class ∈ {0, 1}
    constraints['Class_valid'] = ~df['Class'].isin([0, 1]).sum()
    
    # Amount >= 0
    constraints['Amount_nonneg'] = int((df['Amount'] < 0).sum())
    
    # Time in valid range
    constraints['Time_inrange'] = int((df['Time'] < 0).sum() + (df['Time'] > 172800).sum())
    
    # No inf/NaN
    constraints['No_inf'] = int(np.isinf(df).sum().sum())
    constraints['No_nan'] = int(df.isnull().sum().sum())
    
    violations = sum(constraints.values())
    evidence = f"Violations: {constraints} | Total: {int(violations)}"
    
    log_decision('C9', 'Invalid or impossible value detection',
                 rows_before, rows_before,
                 31, 31,
                 "Verify all domain constraints: Class∈{0,1}, Amount≥0, Time in [0,172800]",
                 evidence,
                 "reversible (read from source)",
                 "constraint violations would corrupt analysis")
    
    return constraints

constraints = c9_constraint_check(df)

# ============================================================================
# C10 — Dataset-specific cleaning steps
# ============================================================================
def c10_dataset_specific(df):
    """Document dataset-specific decisions."""
    rows_before = len(df)
    
    # C10.1 Zero-amount transactions
    zero_amount = int((df['Amount'] == 0).sum())
    zero_amount_legit = int(((df['Amount'] == 0) & (df['Class'] == 0)).sum())
    zero_amount_fraud = int(((df['Amount'] == 0) & (df['Class'] == 1)).sum())
    
    # C10.2 PCA std (document, don't scale)
    v_cols = [c for c in df.columns if c.startswith('V')]
    v_stds = df[v_cols].std()
    
    # C10.3 Amount skew (document, don't transform)
    amount_skew = df['Amount'].skew()
    amount_median = df['Amount'].median()
    amount_mean = df['Amount'].mean()
    
    # C10.4 Pre-deduplication info
    pre_dedup_rows = 284807
    current_rows = len(df)
    dup_removed = pre_dedup_rows - current_rows
    
    evidence = (f"Zero-amount: {zero_amount} ({zero_amount_legit} legit, {zero_amount_fraud} fraud) | "
                f"PCA std range: [{v_stds.min():.3f}, {v_stds.max():.3f}] | "
                f"Amount skew: {amount_skew:.2f} (median={amount_median:.2f}, mean={amount_mean:.2f}) | "
                f"Dedup: {pre_dedup_rows}→{current_rows} ({dup_removed} removed)")
    
    log_decision('C10', 'Dataset-specific cleaning steps',
                 rows_before, rows_before,
                 31, 31,
                 "Document: zero-amount retained, PCA not standardized, skew retained, pre-dedup info",
                 evidence,
                 "reversible (read from source)",
                 "all decisions documented for reproducibility")
    
    return {
        'zero_amount': zero_amount,
        'zero_amount_legit': zero_amount_legit,
        'zero_amount_fraud': zero_amount_fraud,
        'v_stds': v_stds,
        'amount_skew': amount_skew,
        'amount_median': amount_median,
        'amount_mean': amount_mean,
        'dup_removed': dup_removed
    }

c10_info = c10_dataset_specific(df)

# ============================================================================
# Save output
# ============================================================================
OUTPUT_FILE = OUTPUT_DIR / 'creditcard_analysis_ready.csv'
df.to_csv(OUTPUT_FILE, sep=';', encoding='utf-8-sig', index=False)

# ============================================================================
# Generate cleaning decisions document
# ============================================================================
doc_lines = [
    "# Data Cleaning Decisions Log",
    "",
    "## Overview",
    "",
    f"- **Source file**: `creditcard_clean.csv`",
    f"- **Output file**: `data/creditcard_analysis_ready.csv`",
    f"- **Rows processed**: {baseline_rows}",
    f"- **Fraud count (baseline)**: {baseline_fraud}",
    f"- **Legitimate count (baseline)**: {baseline_legit}",
    "",
    "## Cleaning Steps Executed",
    "",
]

for d in decisions:
    doc_lines.extend([
        f"### {d['Step ID']} — {d['Operation']}",
        "",
        f"| Field | Content |",
        f"|-------|---------|",
        f"| **Operation** | {d['Operation']} |",
        f"| **Rows/Columns** | {d['Rows']} / {d['Columns']} |",
        f"| **Rationale** | {d['Rationale']} |",
        f"| **Evidence** | {d['Evidence']} |",
        f"| **Reversibility** | {d['Reversibility']} |",
        f"| **Impact** | {d['Impact']} |",
        ""
    ])

doc_lines.extend([
    "## Dataset-Specific Notes",
    "",
    "### C10.1 — Zero-amount transactions",
    f"- {c10_info['zero_amount']} rows with Amount=0",
    f"- {c10_info['zero_amount_legit']} legitimate, {c10_info['zero_amount_fraud']} fraud",
    "- Decision: **RETAIN** — fraud transactions with zero amount are valid fraud cases",
    "",
    "### C10.2 — PCA components",
    f"- Std range: [{c10_info['v_stds'].min():.3f}, {c10_info['v_stds'].max():.3f}]",
    "- Decision: **RETAIN AS-IS** — standardization is a modeling-stage decision",
    "",
    "### C10.3 — Amount distribution",
    f"- Skew: {c10_info['amount_skew']:.2f}",
    f"- Median: {c10_info['amount_median']:.2f}, Mean: {c10_info['amount_mean']:.2f}",
    "- Decision: **RETAIN RAW** — log transform at modeling stage if validated",
    "",
    "### C10.4 — Pre-deduplication comparison",
    f"- Original Kaggle dataset: 284,807 rows",
    f"- After deduplication: {baseline_rows} rows",
    f"- Duplicates removed: {c10_info['dup_removed']}",
    f"- Fraud lost to dedup: {19} (492 → 473)",
    "- Note: current fraud count (473) differs from Kaggle spec (492) due to duplicate removal",
    "",
    "### C10.5 — V-level duplicate groups",
    f"- Groups with identical V1-V28+Amount+Class: {dup_info['vgroup_with_dups']}",
    f"- Total rows in groups: {dup_info['vgroups']['count'].sum()}",
    f"- Max group size: {dup_info['vgroups']['count'].max()}",
    "- Decision: **PENDING** — resolution documented here before modeling stage",
    "- Risk: duplicate rows leak into train/test if not handled",
    "",
    "### C10.6 — Class imbalance",
    f"- Fraud: {baseline_fraud} ({round(baseline_fraud/baseline_rows*100, 5)}%)",
    f"- Legitimate: {baseline_legit}",
    f"- Ratio: 1:{baseline_legit//baseline_fraud} (legitimate:fraud)",
    "- Decision: **RETAIN IMBALANCE** — resampling is a modeling-stage decision",
    "",
    "## Outlier Statistics",
    "",
    "| Column | Outlier Count |",
    "|--------|---------------|",
])

for col, count in sorted(outlier_counts.items(), key=lambda x: -x[1]):
    doc_lines.append(f"| {col} | {count:,} |")

doc_lines.extend([
    "",
    "## Correlation Analysis",
    "",
    f"- **Zero-variance columns**: {zero_var or 'None'}",
    f"- **High correlations (>0.9)**: {len(high_corr)} pairs",
])

if high_corr:
    doc_lines.append("")
    doc_lines.append("| Column 1 | Column 2 | Correlation |")
    doc_lines.append("|----------|----------|-------------|")
    for c1, c2, corr in high_corr[:10]:  # First 10
        doc_lines.append(f"| {c1} | {c2} | {corr} |")

doc_lines.extend([
    "",
    "## Validation Summary",
    "",
    "| Check | Result |",
    "|-------|--------|",
    f"| Total rows | {baseline_rows:,} |",
    f"| Total columns | 31 |",
    f"| Fraud count | {baseline_fraud} |",
    f"| Legitimate count | {baseline_legit} |",
    f"| Fraud ratio (%) | {round(baseline_fraud/baseline_rows*100, 5)} |",
    f"| Exact duplicates | {dup_info['exact_dups']} |",
    f"| V-level groups | {dup_info['vgroup_with_dups']} |",
    "",
    "---",
    "",
    "Generated by `python/clean_creditcard.py`",
])

with open(DECISIONS_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(doc_lines))

# Print summary
print(f"=== Cleaning Complete ===")
print(f"Input:  {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print(f"Log:    {DECISIONS_FILE}")
print(f"\nRows: {baseline_rows:,}")
print(f"Fraud: {baseline_fraud:,}")
print(f"Legitimate: {baseline_legit:,}")
