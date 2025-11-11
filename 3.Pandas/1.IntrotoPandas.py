import pandas as pd
import numpy as np

"""
Intro to pandas — theoretical overview + small examples

This file provides a concise theoretical introduction to the pandas library
and minimal runnable examples demonstrating typical workflows.

Pandas purpose (short):
- pandas provides high-level data structures (Series, DataFrame) for
    labeled, tabular and time-series data.
- It focuses on fast, flexible and expressive data manipulation,
    aggregation, cleaning and I/O for structured data.
- Built on top of NumPy for numeric arrays; integrates with many
    Python data ecosystem tools (matplotlib, scikit-learn, SQL connectors).

Core concepts:
- Series: 1D labeled array (values + index).
- DataFrame: 2D tabular structure of Series sharing a common index.
- Index: immutable labels for rows (and columns) supporting alignment.
- Vectorized operations: elementwise ops operate across arrays efficiently.
- Missing data handling: pandas uses NaN/NaT and offers dropna/fillna.
- GroupBy: split-apply-combine for aggregation and transformation.
- Merge/Join/Concat: relational operations between tables.
- I/O: read/write from CSV, Excel, SQL, JSON, parquet, etc.

Common operations (examples below):
- creation, inspection (head(), info(), describe()), selection (loc/iloc),
    boolean indexing, groupby/agg, handling missing values, merges, reading/writing.

Run this file as a demonstration. It requires pandas (and optionally matplotlib).
"""


# --- Creation ---
# Series: 1D labeled array
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'], name='example_series')

# DataFrame: from dict of lists or list of dicts
df = pd.DataFrame({
        'id': [1, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
        'age': [25, 30, np.nan, 40],
        'score': [85.0, 92.5, 78.0, 88.5],
        'group': ['A', 'B', 'A', 'B']
})

# --- Inspection ---
print("=== head ===")
print(df.head())

print("\n=== info ===")
print(df.info())

print("\n=== describe ===")
print(df.describe(include='all'))

# --- Selection / Indexing ---
# single column -> Series
ages = df['age']
# multiple columns -> DataFrame
subset = df[['name', 'score']]

# loc: label-based (rows by index values, columns by names)
# iloc: integer position based
first_two_rows = df.iloc[0:2]
row_with_id_3 = df.loc[df['id'] == 3]

print("\n=== selection examples ===")
print("ages:\n", ages)
print("subset:\n", subset)
print("first_two_rows:\n", first_two_rows)
print("row_with_id_3:\n", row_with_id_3)

# boolean indexing
high_scores = df[df['score'] > 88]
print("\n=== high_scores ===")
print(high_scores)

# --- Missing data handling ---
print("\n=== missing values ===")
print("isnull:\n", df.isnull())
# drop rows with any missing values
df_dropna = df.dropna()
# fill missing age with median
median_age = df['age'].median()
df_filled = df.fillna({'age': median_age})

print("dropna result:\n", df_dropna)
print("filled missing age:\n", df_filled)

# --- GroupBy / Aggregation ---
grouped = df.groupby('group').agg({
        'score': ['mean', 'min', 'max'],
        'age': 'median'
})
print("\n=== grouped aggregate ===")
print(grouped)

# --- Merge / Concat ---
other = pd.DataFrame({
        'id': [3, 4, 5],
        'registered': [True, False, True]
})
merged = pd.merge(df, other, on='id', how='left')  # left join keeps all df rows
print("\n=== merged ===")
print(merged)

# vertical concat
df_extra = pd.DataFrame({
        'id': [5],
        'name': ['Eve'],
        'age': [28],
        'score': [90.0],
        'group': ['A']
})
concatenated = pd.concat([df, df_extra], ignore_index=True)
print("\n=== concatenated ===")
print(concatenated)

# --- Apply / Transform ---
# elementwise with vectorized ops is preferred; use apply for row-wise custom logic
concatenated['passed'] = concatenated['score'] >= 80
# custom row-wise: create a label
def label_row(row):
        if pd.isna(row['age']):
                return 'unknown'
        return 'young' if row['age'] < 30 else 'adult'
concatenated['age_label'] = concatenated.apply(label_row, axis=1)

print("\n=== apply example ===")
print(concatenated[['name', 'age', 'age_label', 'passed']])

# --- I/O (examples; commented out to avoid accidental file ops) ---
# df.to_csv('output.csv', index=False)
# df_read = pd.read_csv('output.csv')

# --- Performance tips (short) ---
# - Avoid Python loops; use vectorized pandas/NumPy ops.
# - Prefer categorical dtype for repeated string categories.
# - Use chunksize or dask for very large files that don't fit memory.
# - Use inplace operations cautiously (pandas may return copies).

# End of demonstration
if __name__ == "__main__":
        print("\nDemo finished. This script showed a concise theoretical intro and examples for pandas.")