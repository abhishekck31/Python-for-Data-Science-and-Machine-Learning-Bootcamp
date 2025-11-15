import pandas as pd
import numpy as np

"""
Introtoseries.py

Concise notes and runnable examples for pandas.Series (intro level).

Topics:
- Creation from list, dict, scalar, numpy
- Indexing, slicing, label vs positional
- Vectorized ops and automatic index alignment
- Boolean indexing / masks
- Missing data handling (isnull, dropna, fillna)
- Series attributes and common methods
- Conversion to/from DataFrame
"""


# --- Creation ---
s_from_list = pd.Series([10, 20, 30], index=["a", "b", "c"])
s_from_dict = pd.Series({"a": 1, "b": 2, "d": 4})
s_scalar = pd.Series(5, index=["x", "y", "z"])           # broadcast scalar
s_numpy = pd.Series(np.arange(3), index=["p", "q", "r"])

# Notes:
# - Series is 1D labeled array: values + index
# - Index can be any hashable labels (strings, ints, dates, ...)

# --- Access / Indexing ---
val_a = s_from_list["a"]          # label-based
val_pos = s_from_list.iloc[1]     # position-based
slice_label = s_from_list["a":"b"]  # inclusive for labels
slice_pos = s_from_list.iloc[0:2]   # half-open for positions

# --- Vectorized operations & alignment ---
# Operations are elementwise; when indexes differ, alignment occurs and missing values appear
sum_series = s_from_list + s_from_dict
# For arithmetic ignoring alignment, convert to numpy: s_from_list.values + s_numpy.values

# --- Boolean indexing / masks ---
mask = s_from_list > 15
filtered = s_from_list[mask]
# chaining: s_from_list[s_from_list % 20 == 0]

# --- Missing data ---
s_with_nan = pd.Series([1, np.nan, 3], index=["a", "b", "c"])
nulls = s_with_nan.isnull()
not_nulls = s_with_nan.dropna()
filled = s_with_nan.fillna(0)

# --- Reindexing ---
s_reindexed = s_from_list.reindex(["a", "b", "c", "d"])   # introduces NaN for missing labels
s_reindexed.fillna(method="ffill", inplace=False)         # forward-fill example

# --- Attributes & common methods ---
name = s_from_list.name              # often None
dtype = s_from_list.dtype
index = s_from_list.index
values = s_from_list.values
length = len(s_from_list)

methods_example = {
    "sum": s_from_list.sum(),
    "mean": s_from_list.mean(),
    "describe": s_from_list.describe().to_dict(),
    "unique (from numpy)": np.unique(s_from_list.values).tolist()
}

# --- Series <-> DataFrame ---
df = s_from_list.to_frame(name="col1")   # Series -> DataFrame
s_back = df["col1"]                      # select Series from DataFrame

# --- As a mapping ---
# Series behaves like a dict for lookups; supports get with default
v = s_from_dict.get("c", "missing")

# --- Example run prints ---
if __name__ == "__main__":
    print("s_from_list:\n", s_from_list, "\n")
    print("s_from_dict:\n", s_from_dict, "\n")
    print("sum_series (alignment):\n", sum_series, "\n")
    print("filtered (>15):\n", filtered, "\n")
    print("s_with_nan isnull:\n", nulls, "\n")
    print("filled:\n", filled, "\n")
    print("reindexed:\n", s_reindexed, "\n")
    print("methods_example:\n", methods_example, "\n")
    print("df from series:\n", df, "\n")
    print("v (get with default for 'c'):", v)