import pandas as pd

df = pd.read_csv('D:/Kuliah/Smt 5/Data Science/P2/creditcard.csv')

rows = []

rows.append({'Check': 'Total rows',           'Expected': '284807',   'Value': str(df.shape[0]),                     'Status': 'PASS' if df.shape[0] == 284807 else 'FAIL'})
rows.append({'Check': 'Total columns',        'Expected': '31',       'Value': str(df.shape[1]),                     'Status': 'PASS' if df.shape[1] == 31 else 'FAIL'})

null_total = int(df.isnull().sum().sum())
rows.append({'Check': 'Null values total',    'Expected': '0',        'Value': str(null_total),                      'Status': 'PASS' if null_total == 0 else 'FAIL'})

fraud = int((df['Class'] == 1).sum())
legit = int((df['Class'] == 0).sum())
ratio = round(df['Class'].mean() * 100, 4)
rows.append({'Check': 'Fraud count',          'Expected': '492',      'Value': str(fraud),                           'Status': 'PASS' if fraud == 492 else 'FAIL'})
rows.append({'Check': 'Legitimate count',     'Expected': '284315',   'Value': str(legit),                           'Status': 'PASS' if legit == 284315 else 'FAIL'})
rows.append({'Check': 'Fraud ratio (%)',      'Expected': '~0.172',   'Value': str(ratio),                           'Status': 'PASS' if 0.17 < ratio < 0.18 else 'FAIL'})

rows.append({'Check': 'Time min',             'Expected': '0',        'Value': str(df['Time'].min()),                'Status': 'PASS' if df['Time'].min() == 0 else 'FAIL'})
rows.append({'Check': 'Time max (<=172800s)', 'Expected': '<=172800', 'Value': str(df['Time'].max()),                'Status': 'PASS' if df['Time'].max() <= 172800 else 'FAIL'})

neg_amount = int((df['Amount'] < 0).sum())
rows.append({'Check': 'Negative Amount rows', 'Expected': '0',        'Value': str(neg_amount),                      'Status': 'PASS' if neg_amount == 0 else 'FAIL'})

invalid_class = int((~df['Class'].isin([0, 1])).sum())
rows.append({'Check': 'Class values only 0/1','Expected': '0 invalid','Value': str(invalid_class),                   'Status': 'PASS' if invalid_class == 0 else 'FAIL'})

vcols = [c for c in df.columns if c.startswith('V')]
pca_max_mean = round(df[vcols].mean().abs().max(), 6)
rows.append({'Check': 'V1-V28 mean ~0 (PCA)', 'Expected': '<0.001',  'Value': str(pca_max_mean),                    'Status': 'PASS' if pca_max_mean < 0.001 else 'FAIL'})

dups = int(df.duplicated().sum())
rows.append({'Check': 'Duplicate rows',       'Expected': '0 (ideal)','Value': str(dups),                           'Status': 'WARN' if dups > 0 else 'PASS'})

out = pd.DataFrame(rows)
out.to_csv('D:/Kuliah/Smt 5/Data Science/P2/creditcard_validation_report.csv', sep=';', encoding='utf-8-sig', index=False)

print(out.to_string(index=False))
print('\nSaved: D:/Kuliah/Smt 5/Data Science/P2/creditcard_validation_report.csv')
