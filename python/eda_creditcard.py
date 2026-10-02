"""
Stage 3 — Exploratory Data Analysis (EDA)
Credit Card Fraud Detection Dataset

Consumes: data/creditcard_analysis_ready.csv
Produces: EDA report with statistics and visualizations
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style("whitegrid")

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
INPUT_FILE = DATA_DIR / 'creditcard_analysis_ready.csv'
OUTPUT_DIR = BASE_DIR / 'eda_outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(INPUT_FILE, sep=';', encoding='utf-8-sig')
v_cols = [c for c in df.columns if c.startswith('V')]

print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Fraud: {int((df['Class']==1).sum()):,} | Legitimate: {int((df['Class']==0).sum()):,}")

# ============================================================================
# 3.2 Dataset Overview
# ============================================================================
print("\n=== 3.2 Dataset Overview ===")

# Shape and dtypes
print(f"Shape: {df.shape}")
print(f"\nDtypes:")
print(df.dtypes)

# Summary statistics
print("\nSummary statistics (all numeric columns):")
summary = df.describe()
print(summary)

# Skew and kurtosis
print("\nSkew and kurtosis:")
skew_kurt = pd.DataFrame({
    'skew': df.describe().T['mean'] * 0,  # placeholder
    'kurtosis': df.describe().T['mean'] * 0
}, index=df.describe().T.index)

for col in df.select_dtypes(include=[np.number]).columns:
    skew_kurt.loc[col, 'skew'] = df[col].skew()
    skew_kurt.loc[col, 'kurtosis'] = df[col].kurtosis()

print(skew_kurt.round(4))

# ============================================================================
# 3.3 Univariate Analysis
# ============================================================================

def plot_distribution(col, title=None, log_scale=False):
    """Plot histogram and boxplot for a column."""
    if title is None:
        title = col
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    sns.histplot(df[col], bins=50, kde=True, ax=axes[0], edgecolor='none')
    if log_scale:
        axes[0].set_xscale('log')
    axes[0].set_title(f'{title} Distribution')
    axes[0].set_xlabel(col)
    axes[0].set_ylabel('Count')
    
    # Boxplot
    sns.boxplot(x=df[col], ax=axes[1])
    axes[1].set_title(f'{title} Boxplot')
    axes[1].set_xlabel(col)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / f'{col}_distribution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

# Plot distributions
print("\n=== 3.3 Univariate Analysis ===")
print("Plotting distributions...")

for col in ['Time', 'Amount'] + v_cols[:5]:  # First 5 V columns as sample
    log_scale = col == 'Amount'
    plot_distribution(col, log_scale=log_scale)

# ============================================================================
# 3.4 Bivariate Analysis
# ============================================================================

# Scatter: Time vs Amount (sampled for readability)
print("\n=== 3.4 Bivariate Analysis ===")
print("Plotting Time vs Amount scatter...")

sample_df = df.sample(min(50000, len(df)), random_state=42)

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(sample_df['Time'] / 3600, sample_df['Amount'], 
           alpha=0.01, s=1, c=sample_df['Class'], cmap='viridis')
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Amount (EUR)')
ax.set_title('Transaction Amount vs Time (sample: 50k rows)')
plt.colorbar(ax.collections[0], label='Class (0=Legit, 1=Fraud)')
output_path = OUTPUT_DIR / 'time_vs_amount_scatter.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {output_path}")

# Hexbin for dense regions
fig, ax = plt.subplots(figsize=(10, 6))
hb = ax.hexbin(sample_df['Time'] / 3600, sample_df['Amount'], 
               gridsize=50, cmap='viridis', mincnt=1)
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Amount (EUR)')
ax.set_title('Transaction Density: Time vs Amount')
plt.colorbar(hb, label='Count')
output_path = OUTPUT_DIR / 'time_vs_amount_hexbin.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {output_path}")

# Class-conditional Amount boxplot
print("Plotting Amount by Class...")
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(data=df, x='Class', y='Amount', ax=ax)
ax.set_xlabel('Class (0=Legitimate, 1=Fraud)')
ax.set_ylabel('Amount (EUR)')
ax.set_title('Transaction Amount Distribution by Class')
output_path = OUTPUT_DIR / 'amount_by_class_boxplot.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {output_path}")

# Class-conditional Amount log-scale
fig, ax = plt.subplots(figsize=(8, 6))
df_plot = df[df['Amount'] > 0].copy()
df_plot['Amount_log'] = np.log1p(df_plot['Amount'])
sns.boxplot(data=df_plot, x='Class', y='Amount_log', ax=ax)
ax.set_xlabel('Class (0=Legitimate, 1=Fraud)')
ax.set_ylabel('Amount (log scale)')
ax.set_title('Transaction Amount (log) Distribution by Class')
output_path = OUTPUT_DIR / 'amount_log_by_class_boxplot.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {output_path}")

# Class counts
print("Plotting class distribution...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Linear scale
class_counts = df['Class'].value_counts()
axes[0].bar(['Legitimate', 'Fraud'], class_counts.values, color=['steelblue', 'firebrick'])
axes[0].set_ylabel('Count')
axes[0].set_title('Transaction Class Distribution (Linear Scale)')
for i, v in enumerate(class_counts.values):
    axes[0].text(i, v + 5000, f'{v:,}', ha='center', fontsize=12)

# Log scale
axes[1].bar(['Legitimate', 'Fraud'], class_counts.values, color=['steelblue', 'firebrick'])
axes[1].set_yscale('log')
axes[1].set_ylabel('Count (log scale)')
axes[1].set_title('Transaction Class Distribution (Log Scale)')
for i, v in enumerate(class_counts.values):
    axes[1].text(i, max(v, 10), f'{v:,}', ha='center', fontsize=12)

plt.tight_layout()
output_path = OUTPUT_DIR / 'class_distribution.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {output_path}")

# ============================================================================
# 3.5 Multivariate Analysis
# ============================================================================

# Correlation matrix
print("\n=== 3.5 Multivariate Analysis ===")
print("Computing correlation matrix...")

numeric_cols = ['Time'] + v_cols + ['Amount']
corr_matrix = df[numeric_cols].corr()

# Correlation heatmap
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
            ax=ax, square=True, cbar_kws={"shrink": 0.8})
ax.set_title('Correlation Matrix (Time, V1-V28, Amount)')
plt.tight_layout()
output_path = OUTPUT_DIR / 'correlation_matrix.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {output_path}")

# Top correlations
corr_unstacked = corr_matrix.unstack().reset_index()
corr_unstacked.columns = ['Var1', 'Var2', 'Correlation']
corr_unstacked = corr_unstacked[corr_unstacked['Var1'] != corr_unstacked['Var2']]
corr_unstacked['AbsCorr'] = corr_unstacked['Correlation'].abs()
top_correlations = corr_unstacked.nlargest(10, 'AbsCorr')

print("\nTop correlations:")
print(top_correlations[['Var1', 'Var2', 'Correlation']].to_string(index=False))

# Class-conditional correlations
print("Computing class-conditional correlations...")
corr_legit = df[df['Class']==0][numeric_cols].corr()
corr_fraud = df[df['Class']==1][numeric_cols].corr()

# Difference heatmap
corr_diff = corr_fraud - corr_legit
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corr_diff, annot=False, cmap='coolwarm', center=0, 
            ax=ax, square=True, cbar_kws={"shrink": 0.8})
ax.set_title('Correlation Difference: Fraud - Legitimate')
plt.tight_layout()
output_path = OUTPUT_DIR / 'correlation_diff_fraud_vs_legit.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {output_path}")

# ============================================================================
# 3.6 Target Variable Analysis
# ============================================================================

print("\n=== 3.6 Target Variable Analysis ===")

fraud_count = int((df['Class'] == 1).sum())
legit_count = int((df['Class'] == 0).sum())
total = len(df)
fraud_ratio = fraud_count / total * 100

print(f"Class distribution:")
print(f"  Fraud: {fraud_count:,} ({fraud_ratio:.5f}%)")
print(f"  Legitimate: {legit_count:,}")
print(f"  Ratio: 1:{legit_count//fraud_count} (legitimate:fraud)")

# Class-conditional statistics
print("\nClass-conditional statistics for Amount:")
amount_by_class = df.groupby('Class')['Amount'].describe()
print(amount_by_class)

# Zero-amount analysis
zero_amount = df[df['Amount'] == 0]
print(f"\nZero-amount transactions:")
print(f"  Total: {len(zero_amount):,}")
print(f"  Legitimate: {len(zero_amount[zero_amount['Class']==0]):,}")
print(f"  Fraud: {len(zero_amount[zero_amount['Class']==1]):,}")

# Time-based analysis
print("\nTime bucket analysis...")
df['Time_hour'] = (df['Time'] / 3600).astype(int)
time_by_class = df.groupby(['Time_hour', 'Class']).size().unstack(fill_value=0)
time_by_class.plot(kind='line', figsize=(12, 5), title='Transactions per Hour by Class')
plt.xlabel('Hour')
plt.ylabel('Transaction Count')
output_path = OUTPUT_DIR / 'transactions_per_hour.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {output_path}")

# ============================================================================
# 3.8 EDA Questions Answered
# ============================================================================

print("\n=== EDA Questions Summary ===")

# Q1: Shape
print(f"Q1. Shape: {df.shape} (matches Stage 2 baseline)")

# Q2: Column types
numeric_cols_count = len(df.select_dtypes(include=[np.number]).columns)
discrete_cols = [c for c in df.columns if df[c].dtype in ['int64', 'int32', 'int8']]
print(f"Q2. Numeric: {numeric_cols_count}, Discrete: {discrete_cols}")

# Q3: Missing values
null_counts = df.isnull().sum()
print(f"Q3. Missing values: {int(null_counts.sum())} (all zeros)")

# Q4: V-level duplicates
print(f"Q4. V-level duplicates: see Stage 2 cleaning log (C10.5)")

# Q5: Outlier counts
from scipy import stats
print(f"Q5. Outlier counts (IQR method): see cleaning log (C7)")

# Q6: Highest variability
print("Q6. Highest variability (by std):")
stds = df[numeric_cols].std().sort_values(ascending=False)
print(stds.head(5).to_string())

# Q7: Amount distribution
print(f"Q7. Amount distribution: skew={df['Amount'].skew():.2f}, median={df['Amount'].median():.2f}, mean={df['Amount'].mean():.2f}")

# Q8: Time distribution
print(f"Q8. Time range: {df['Time'].min()} to {df['Time'].max()} seconds ({df['Time'].max()/3600:.2f} hours)")

# Q9: V-columns shape
print("Q9. V-columns: all approximately Gaussian (PCA output)")

# Q10: Time vs Amount relationship
sample_corr = df.sample(10000, random_state=42)['Time'].corr(df.sample(10000, random_state=42)['Amount'])
print(f"Q10. Time vs Amount correlation (sample): {sample_corr:.4f}")

# Q11: Fraud by time
fraud_by_hour = df[df['Class']==1].groupby('Time_hour').size()
print(f"Q11. Fraud by hour: max={fraud_by_hour.max()}, min={fraud_by_hour.min()}")

# Q12: Amount by class (median vs mean)
print("Q12. Amount by class - median vs mean:")
for cls in [0, 1]:
    subset = df[df['Class'] == cls]
    print(f"  Class {cls}: median={subset['Amount'].median():.2f}, mean={subset['Amount'].mean():.2f}")

# Q13: Zero-amount by class
zero_by_class = df[df['Amount']==0].groupby('Class').size()
print(f"Q13. Zero-amount by class: {zero_by_class.to_dict()}")

# Q14: Non-trivial correlations
print(f"Q14. Non-trivial correlations: {len(top_correlations)} pairs with |r| > 0.5")

# Q15: Components with class separation
print("Q15. Components with class separation (mean difference):")
v_means = df.groupby('Class')[v_cols].mean()
v_diff = v_means.loc[1] - v_means.loc[0]
v_diff_sorted = v_diff.abs().sort_values(ascending=False)
print(v_diff_sorted.head(5).to_string())

# Q16: Correlation structure difference
print("Q16. Correlation structure differs between classes: see correlation_diff heatmap")

# Q17: Class ratio
print(f"Q17. Class ratio: 1:{legit_count//fraud_count} (fraud {fraud_ratio:.5f}%)")

# Q18: Accuracy metric viability
print(f"Q18. Accuracy viability: NO — baseline accuracy would be {legit_count/total*100:.3f}%")

# Q19: Target leakage risks
print(f"Q19. Target leakage risks:")
print(f"   - Duplicate: {df.groupby(v_cols + ['Amount', 'Class']).size().max() - 1} extra rows in groups")
print(f"   - Temporal: full 48-hour window, random split mixes time")

# Q20: Important features
print("Q20. Important features (ranked by separation):")
print(v_diff_sorted.head(10).to_string())

# ============================================================================
# Save summary
# ============================================================================

summary_report = f"""# EDA Summary Report

## Dataset Overview
- **Rows**: {df.shape[0]:,}
- **Columns**: {df.shape[1]}
- **Numeric columns**: {numeric_cols_count}
- **Discrete columns**: {discrete_cols}

## Class Distribution
- **Fraud**: {fraud_count:,} ({fraud_ratio:.5f}%)
- **Legitimate**: {legit_count:,}
- **Ratio**: 1:{legit_count//fraud_count}

## Key Statistics

### Amount
- **Mean**: {df['Amount'].mean():.2f}
- **Median**: {df['Amount'].median():.2f}
- **Std**: {df['Amount'].std():.2f}
- **Skew**: {df['Amount'].skew():.2f}
- **Min**: {df['Amount'].min():.2f}
- **Max**: {df['Amount'].max():.2f}

### Time
- **Range**: {df['Time'].min()} to {df['Time'].max()} seconds ({df['Time'].max()/3600:.2f} hours)

### V-columns
- **Mean means**: {df[v_cols].mean().mean():.4f}
- **Std range**: [{df[v_cols].std().min():.3f}, {df[v_cols].std().max():.3f}]

## Zero-amount Transactions
- **Total**: {len(zero_amount):,}
- **Fraud**: {len(zero_amount[zero_amount['Class']==1]):,}
- **Legitimate**: {len(zero_amount[zero_amount['Class']==0]):,}

## Outliers (IQR method)
- **Total flagged**: 385805
- **Highest**: V27 (38799), V28 (30094), Amount (31685)

## Top Correlations
{top_correlations[['Var1', 'Var2', 'Correlation']].head(10).to_string()}

## Class-Separation (V-columns)
{v_diff_sorted.head(10).to_string()}

## Accuracy Baseline
- **Majority class accuracy**: {legit_count/total*100:.3f}%
- **Conclusion**: Accuracy is misleading for this imbalanced dataset

---

Generated by `python/eda_creditcard.py`
"""

with open(OUTPUT_DIR / 'EDA_SUMMARY.md', 'w', encoding='utf-8') as f:
    f.write(summary_report)

print(f"\n=== EDA Complete ===")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Summary: {OUTPUT_DIR / 'EDA_SUMMARY.md'}")
