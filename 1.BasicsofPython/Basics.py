from __future__ import annotations
import sys
import math
import random
from typing import List, Tuple, Dict, Iterator
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

"""
Basics.py
A compact, runnable Python "cheat-sheet" covering the essentials needed to start
with machine learning: core Python, data structures, functions, modules, and
short examples using numpy, pandas, matplotlib, and scikit-learn.

Run this file to see short demos. Missing optional packages will print install tips.
"""


# ---------------------------
# 1. Basic syntax & types
# ---------------------------
print("1. Basic types")
a: int = 10
b: float = 3.14
c: str = "hello"
d: bool = True
e = None

print("int:", a, "float:", b, "str:", c, "bool:", d, "None:", e)

# Type conversion
print("convert:", int(3.9), float("2.5"), str(123))

# Formatted strings
name = "ML"
print(f"Welcome to {name} basics. a={a}, b={b:.2f}")

# ---------------------------
# 2. Collections
# ---------------------------
print("\n2. Collections")

# Lists (mutable)
lst: List[int] = [1, 2, 3]
lst.append(4)
lst[0] = 0
print("list:", lst)

# Tuples (immutable)
tpl: Tuple[int, int] = (1, 2)
print("tuple:", tpl)

# Dictionaries
dct: Dict[str, int] = {"one": 1, "two": 2}
dct["three"] = 3
print("dict:", dct, "keys:", list(dct.keys()))

# Sets
s = {1, 2, 2, 3}
print("set:", s)

# ---------------------------
# 3. Control flow
# ---------------------------
print("\n3. Control flow")

x = 7
if x % 2 == 0:
    print("even")
else:
    print("odd")

for i in range(5):
    print(i, end=" ")
print()

i = 0
while i < 3:
    print("while", i)
    i += 1

# ---------------------------
# 4. Functions, lambdas, comprehensions
# ---------------------------
print("\n4. Functions & functional tools")

def square(n: int) -> int:
    return n * n

print("square(5):", square(5))

# Multiple return values (tuples)
def stats(numbers: List[float]) -> Tuple[float, float]:
    return (sum(numbers) / len(numbers), max(numbers) - min(numbers))

print("stats([1,4,9]):", stats([1, 4, 9]))

# Lambda
add = lambda x, y: x + y
print("lambda add:", add(2, 3))

# List comprehensions
squares = [x * x for x in range(6)]
print("squares:", squares)

# Generator (memory efficient)
def naturals(limit: int) -> Iterator[int]:
    n = 0
    while n < limit:
        yield n
        n += 1

print("naturals:", list(naturals(5)))

# map, filter
print("map:", list(map(square, [1,2,3])))
print("filter evens:", list(filter(lambda z: z % 2 == 0, range(6))))

# ---------------------------
# 5. File I/O (text)
# ---------------------------
print("\n5. File I/O example (creates 'example.txt')")
with open("example.txt", "w", encoding="utf-8") as f:
    f.write("hello\nworld\n123\n")

with open("example.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f]
print("example.txt lines:", lines)

# ---------------------------
# 6. Modules & packages
# ---------------------------
# Use import to bring code from other files or standard library modules.
# Create a virtual environment for ML projects:
# python -m venv venv
# Activate it, then install required packages:
# pip install numpy pandas matplotlib scikit-learn

# ---------------------------
# 7. Numeric computing with numpy
# ---------------------------
print("\n7. numpy basics (optional)")

try:
    import numpy as np
except Exception as exc:
    print("numpy not installed. Install with: pip install numpy")
    np = None

if np is not None:
    a = np.array([1, 2, 3])
    b = np.arange(6).reshape(2, 3)
    print("np array a:", a, "shape:", a.shape)
    print("np array b:\n", b)
    print("elementwise ops:", a + 1, a * 2)
    print("mean:", np.mean(a), "std:", np.std(a))
    # Broadcasting
    print("broadcast:", a + np.array([10, 20, 30]))

# ---------------------------
# 8. pandas basics
print("\n8. pandas basics (optional)")

try:
    import pandas as pd
except Exception as exc:
    print("pandas not installed. Install with: pip install pandas")
    pd = None

if pd is not None:
    df = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0],
        "feature2": [4.0, 5.5, 6.1],
        "label": [0, 1, 0]
    })
    print("DataFrame:\n", df)
    print("describe:\n", df.describe())
    # selecting
    print("feature1 column:", df["feature1"].tolist())
    # reading csv (example)
    df.to_csv("example.csv", index=False)
    print("wrote example.csv")
    print("wrote example.csv")

# ---------------------------
# 9. plotting with matplotlib
# ---------------------------
print("\n9. matplotlib (optional)")

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    print("matplotlib not installed. Install with: pip install matplotlib")
    plt = None

if plt is not None and np is not None:
    x = np.linspace(0, 2 * np.pi, 50)
    y = np.sin(x)
    # Quick plot (will open a window if run locally)
    plt.plot(x, y, label="sin")
    plt.title("Sine wave")
    plt.legend()
    plt.savefig("sine_plot.png")
    plt.close()
    print("Saved sine_plot.png")

# ---------------------------
print("\n10. Simple ML with scikit-learn (optional)")

try:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
except Exception as exc:
    print("scikit-learn not installed. Install with: pip install scikit-learn")
    # Define placeholders to avoid NameError further
    train_test_split = None
    LinearRegression = None
    mean_squared_error = None
    r2_score = None

if np is not None and LinearRegression is not None:
    # Create synthetic linear data with noise
    rng = np.random.default_rng(seed=42)
    X = 2.5 * rng.random(100).reshape(-1, 1)  # 100 samples, single feature
    true_coef = 1.75
    y = (true_coef * X.flatten()) + 0.5 + rng.normal(scale=0.3, size=100)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict & evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Learned coef: {model.coef_[0]:.4f}, intercept: {model.intercept_:.4f}")
    print(f"MSE: {mse:.4f}, R2: {r2:.4f}")

    # Save prediction example to CSV
    try:
        out = pd.DataFrame({"X_test": X_test.flatten(), "y_test": y_test, "y_pred": y_pred})
        out.to_csv("predictions.csv", index=False)
        print("Saved predictions.csv")
    except Exception:
        pass
    except Exception:
        pass

# ---------------------------
# 11. Quick tips & next steps
# ---------------------------
print("\n11. Quick tips")
tips = [
    "Use virtual environments (venv) per project.",
    "Learn numpy arrays and vectorized operations (avoid Python loops for large arrays).",
    "Learn pandas for tabular data manipulation (DataFrame).",
    "Use scikit-learn for standard ML models and pipelines.",
    "Practice by loading datasets (CSV), cleaning data, feature engineering, training and evaluating models."
]
for t in tips:
    print("-", t)

# End of file