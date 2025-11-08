import numpy as np

"""
NumPy arrays - concise notes and examples

File: 2.Numpyarrays.py
Purpose: Quick reference demonstrating common NumPy array operations, attributes,
         indexing/slicing, reshaping, broadcasting, boolean/fancy indexing, and performance tips.
"""


# ---------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------
a = np.array([1, 2, 3])                 # 1D
b = np.array([[1, 2, 3], [4, 5, 6]])    # 2D
c = np.arange(0, 10, 2)                 # [0 2 4 6 8]
d = np.linspace(0, 1, 5)                # 5 evenly spaced points between 0 and 1
z = np.zeros((2, 3), dtype=float)
o = np.ones((2, 2), dtype=int)
I = np.eye(3)                           # identity matrix
r = np.random.default_rng(0).random((3, 3))  # reproducible random numbers

# dtype specification and conversion
arr_int = np.array([1, 2, 3], dtype=np.int32)
arr_float = arr_int.astype(np.float64)

# ---------------------------------------------------------------------
# Attributes
# ---------------------------------------------------------------------
# .ndim, .shape, .size, .dtype, .itemsize, .nbytes
_attrs = {
    "ndim": b.ndim,
    "shape": b.shape,
    "size": b.size,
    "dtype": b.dtype,
    "itemsize": b.itemsize,
    "nbytes": b.nbytes,
}

# ---------------------------------------------------------------------
# Indexing and slicing (like Python lists, extended to multiple dimensions)
# ---------------------------------------------------------------------
x = np.arange(12).reshape(3, 4)  # 3x4: rows x cols
# indexing
elem = x[1, 2]       # second row, third column
row = x[1]           # second row (1D)
col = x[:, 2]        # third column
sub = x[0:2, 1:4]    # slice

# negative indices and steps
last = x[-1, -1]
rev_row = x[0, ::-1]

# ---------------------------------------------------------------------
# Views vs copies
# ---------------------------------------------------------------------
s = x[:, 1:3]        # view (usually) - modifying s may change x
s_copy = s.copy()    # explicit copy
s[0, 0] = 999        # x will reflect this change if s is a view

# flatten vs ravel
flat_copy = x.flatten()  # copy
flat_view = x.ravel()    # view when possible

# ---------------------------------------------------------------------
# Reshaping and transposes
# ---------------------------------------------------------------------
y = np.arange(6)
y2 = y.reshape((2, 3))
y_flat = y.reshape(-1)      # flatten to 1D
t = y2.T                    # transpose for 2D02
z_reshaped = y2.reshape(3, -1)  # infer dimension

# ---------------------------------------------------------------------
# Basic arithmetic and broadcasting
# ---------------------------------------------------------------------
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([10, 20, 30])   # shape (3,)
C = A + B                    # B is broadcasted to shape (2,3)
D = A * 2                    # scalar broadcast
dot = A @ B                  # matrix-vector product (uses broadcasting rules)

# broadcasting rules summary:
# - trailing dimensions must match or be 1
# - arrays are virtually expanded; no data copy unless needed

# ---------------------------------------------------------------------
# Reductions and axis semantics
# ---------------------------------------------------------------------
arr = np.arange(12).reshape(3, 4)
sum_all = arr.sum()
sum_axis0 = arr.sum(axis=0)   # sum along rows => shape (4,)
sum_axis1 = arr.sum(axis=1)   # sum along columns => shape (3,)
mean = arr.mean(axis=1)
max_indices = arr.argmax(axis=1)
min_indices = arr.argmin(axis=0)

# keepdims preserves reduced dimensions with size 1
sum_keep = arr.sum(axis=1, keepdims=True)  # shape (3,1)

# ---------------------------------------------------------------------
# Boolean indexing and masks
# ---------------------------------------------------------------------
vals = np.array([0, 3, -1, 5, 2])
mask = vals > 0
positive = vals[mask]            # selects elements > 0
vals[vals < 0] = 0               # assignment via boolean mask

# np.where: conditional selection with broadcasting
cond = np.where(vals % 2 == 0, "even", "odd")  # returns an array of strings

# ---------------------------------------------------------------------
# Fancy indexing (integer array indexing)
# ---------------------------------------------------------------------
M = np.arange(16).reshape(4, 4)
rows = np.array([0, 2])
cols = np.array([1, 3])
selected = M[rows[:, None], cols]  # pairs: (0,1) and (2,3)

# ---------------------------------------------------------------------
# Stacking and splitting
# ---------------------------------------------------------------------
a1 = np.arange(6).reshape(2, 3)
a2 = np.arange(6, 12).reshape(2, 3)
vstacked = np.vstack([a1, a2])   # shape (4,3)
hstacked = np.hstack([a1, a2])   # shape (2,6)
conc = np.concatenate([a1, a2], axis=0)
split0, split1 = np.split(vstacked, 2, axis=0)

# ---------------------------------------------------------------------
# Advanced: broadcast_to, tile, repeat, einsum
# ---------------------------------------------------------------------
big = np.broadcast_to(np.array([1, 2, 3]), (4, 3))  # no copy, read-only view
tiled = np.tile(np.array([1, 2]), (3,))             # repeat sequence
repeated = np.repeat(np.array([1, 2, 3]), 2)       # repeat elements
eins = np.einsum('ij,jk->ik', A, A.T)               # flexible contractions

# ---------------------------------------------------------------------
# Sorting and unique
# ---------------------------------------------------------------------
unsorted = np.array([3, 1, 2])
sorted_arr = np.sort(unsorted)
idx_sorted = np.argsort(unsorted)
unique_vals = np.unique(np.array([1, 2, 2, 3, 1]))

# ---------------------------------------------------------------------
# Memory layout and performance tips
# ---------------------------------------------------------------------
# - Prefer vectorized ops (avoid Python loops).
# - Use contiguous arrays (C-order) for best performance; check .flags['C_CONTIGUOUS'].
# - Use np.dot or @ for linear algebra, not manual loops.
# - Use appropriate dtype to save memory (e.g., float32 vs float64).
# - Use in-place operations when possible: arr *= 2
# - Use numba or C-extension for heavy custom loops (if needed).

# Example benchmark hint:
# %timeit (arr * 2)           # vectorized
# %timeit ([x * 2 for x in arr])  # Python loop (much slower)

# ---------------------------------------------------------------------
# Small demonstration print (can be removed)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("Example arrays and shapes:")
    print("b.shape =", b.shape, "b.dtype =", b.dtype)
    print("x =\n", x)
    print("s (view) =\n", s)
    print("s_copy (copy) =\n", s_copy)
    print("Broadcast example A+B =\n", C)
    print("Boolean mask positive =", positive)
    print("Fancy indexing selected =\n", selected)
    print("Einsum example (A @ A.T) =\n", eins)