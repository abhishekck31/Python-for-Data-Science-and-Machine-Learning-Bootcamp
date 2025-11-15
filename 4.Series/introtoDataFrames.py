import pandas as pd
import numpy as np

"""
introtoDataFrames.py

Notes + small examples: Introduction to pandas DataFrame

Contents:
- What is a DataFrame
- Creating DataFrames
- Inspecting data (head, info, describe)
- Indexing & selection (.loc, .iloc, boolean masks)
- Adding / dropping columns
- GroupBy, aggregation
- Merge / join
- IO (read_csv / to_csv)
- Missing data handling
- Pivot table and reshaping
- Useful tips (vectorized ops, method chaining)

This file contains commented notes and runnable minimal examples.
"""


# -----------------------
# What is a DataFrame?
# -----------------------
# A DataFrame is a 2D labeled data structure with rows and columns (like a spreadsheet or SQL table).
# Each column is a Series. DataFrames are the primary data structure for tabular data in pandas.

# -----------------------
# Creating DataFrames
# -----------------------
df = pd.DataFrame({
    "id": [101, 102, 103, 104],
    "name": ["Alice", "Bob", "Charlie", "Diana"],
    "age": [25, 32, np.nan, 28],
    "score": [85.5, 92.0, 88.0, 79.5],
    "group": ["A", "B", "A", "B"]
})
print("=== df ===")
print(df)

# -----------------------
# Inspecting data
# -----------------------
print("\nshape:", df.shape)          # (rows, columns)
print("columns:", df.columns.tolist())
print("dtypes:\n", df.dtypes)
print("\nhead:\n", df.head())        # first rows
print("\ninfo:")
df.info()
print("\ndescribe (numeric):\n", df.describe())

# -----------------------
# Indexing & selection
# -----------------------
# By label
print("\n.loc selecting row where id == 102:")
print(df.loc[df['id'] == 102])

# By position
print("\n.iloc first two rows, first three columns:")
print(df.iloc[0:2, 0:3])

# Boolean filtering
older_than_26 = df['age'] > 26
print("\nFiltered age > 26 (note: NaN comparisons yield False):")
print(df[older_than_26])

# Select columns
print("\nSelect columns ['name', 'score']:\n", df[['name', 'score']])

# -----------------------
# Adding / dropping columns
# -----------------------
df['passed'] = df['score'] >= 80
print("\nAdded 'passed' column:\n", df)

# Derived column (vectorized)
df['age_plus_five'] = df['age'] + 5
print("\nAdded 'age_plus_five' (NaN-safe propagation):\n", df)

# Drop column (inplace=False by default)
dropped = df.drop(columns=['age_plus_five'])
print("\nAfter dropping 'age_plus_five' (new object):\n", dropped)

# -----------------------
# Handling missing data
# -----------------------
print("\nMissing values count per column:\n", df.isna().sum())
# Fill missing ages with median
median_age = df['age'].median()
df_filled = df.fillna({'age': median_age})
print("\nFilled NaN age with median (new df):\n", df_filled)

# You can also drop rows with missing values:
df_dropped_na = df.dropna(subset=['age'])
print("\nRows with non-missing age:\n", df_dropped_na)

# -----------------------
# GroupBy and aggregation
# -----------------------
grouped = df.groupby('group').agg(
    count=('id', 'size'),
    avg_score=('score', 'mean'),
    max_age=('age', 'max')
).reset_index()
print("\nGroupBy aggregation:\n", grouped)

# -----------------------
# Merge / join
# -----------------------
df_meta = pd.DataFrame({
    "id": [101, 102, 103, 104],
    "city": ["NY", "LA", "SF", "CHI"]
})
merged = df.merge(df_meta, on='id', how='left')
print("\nMerged df with df_meta:\n", merged)

# -----------------------
# Reshaping: pivot_table
# -----------------------
pivot = merged.pivot_table(index='group', values='score', aggfunc=['mean', 'count'])
print("\nPivot table (score by group):\n", pivot)

# -----------------------
# IO (reading / writing)
# -----------------------
# To read from CSV:
# df_from_csv = pd.read_csv("data.csv")
# To write:
# df.to_csv("out.csv", index=False)

# -----------------------
# Apply / applymap / map
# -----------------------
# Elementwise with applymap (for entire DataFrame) or Series.map for mapping values
df['name_upper'] = df['name'].map(str.upper)
print("\nMapped 'name' to uppercase:\n", df[['name', 'name_upper']])

# -----------------------
# Useful tips
# -----------------------
# - Prefer vectorized ops (fast) over Python loops.
# - Chain operations for readable pipelines:
#   df.pipe(...).query(...).assign(...).groupby(...).
# - Use .loc/.iloc for explicit selection to avoid ambiguous behavior.
# - Use copy() when you need an explicit copy to avoid SettingWithCopyWarning.

if __name__ == "__main__":
    # Minimal demonstration already printed above when file is run.
    pass