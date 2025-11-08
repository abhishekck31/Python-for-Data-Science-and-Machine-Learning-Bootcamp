import numpy as np

def basic_indexing():
    # 1D array indexing and slicing
    a = np.arange(10)           # [0 1 2 3 4 5 6 7 8 9]
    print("a:", a)
    print("a[3]:", a[3])        # single element
    print("a[-1]:", a[-1])      # last element
    print("a[2:7]:", a[2:7])    # slice (view)
    print("a[:5]:", a[:5])
    print("a[::2]:", a[::2])    # stride

def multi_dim_indexing():
    # 2D array indexing
    m = np.arange(1, 13).reshape(3, 4)
    print("\nm:\n", m)
    print("m[1,2]:", m[1, 2])         # row 1, col 2
    print("m[1]:", m[1])               # row 1 (1D slice)
    print("m[:, 2]:", m[:, 2])         # entire column (1D view)
    print("m[0:2, 1:3]:\n", m[0:2, 1:3]) # submatrix (view)

def boolean_indexing():
    # Boolean masking
    x = np.array([10, 15, 20, 25, 30])
    mask = x > 18
    print("\nx:", x)
    print("mask:", mask)
    print("x[mask]:", x[mask])             # elements > 18
    print("x[x % 20 == 0]:", x[x % 20 == 0]) # compound condition

def fancy_indexing():
    # Integer (fancy) indexing returns a copy
    a = np.arange(12).reshape(3, 4)
    rows = np.array([0, 2])
    cols = np.array([1, 3])
    print("\na:\n", a)
    print("a[rows, cols]:", a[rows, cols])    # pairs (0,1) and (2,3)
    # Cross-indexing using ix_ to form a grid
    print("a[np.ix_(rows, cols)]:\n", a[np.ix_(rows, cols)])  # 2x2 submatrix

def advanced_tools():
    a = np.arange(8)
    # take and put (indexing helpers)
    print("\na:", a)
    print("np.take(a, [0,3,6]):", np.take(a, [0, 3, 6]))
    b = a.copy()
    np.put(b, [0, 3], [99, 100])
    print("after np.put:", b)
    # where: indices or conditional selection
    idx = np.where(a % 2 == 0)
    print("where even indices:", idx, "values:", a[idx])

def indexing_and_views_vs_copies():
    a = np.arange(9)
    s = a[2:7]              # view
    s[0] = 999
    print("\na after modifying slice (view):", a)
    # Fancy indexing produces a copy
    a = np.arange(9)
    f = a[[2, 3, 4]]        # copy
    f[0] = -1
    print("a after modifying fancy-index copy:", a)

def special_indexers():
    a = np.arange(24).reshape(2, 3, 4)
    print("\nshape:", a.shape)
    # Ellipsis to skip axes
    print("a[..., 1]:\n", a[..., 1])   # pick index 1 on last axis
    # New axis / None to add dims
    v = np.array([1, 2, 3])
    print("v[:, None].shape:", v[:, None].shape)  # column vector (3,1)
    print("v[None, :].shape:", v[None, :].shape)  # row vector (1,3)

def examples_summary():
    print("\nSummary notes:")
    print("- Basic slices return views when possible.")
    print("- Fancy/integer indexing returns copies.")
    print("- Boolean masks select elements; shape of mask must match indexed axes.")
    print("- Use np.ix_ to build broadcastable indexers for multidim selection.")
    print("- Ellipsis (...) fills unspecified dimensions.")
    print("- None (np.newaxis) adds new axes for broadcasting.")

if __name__ == "__main__":
    basic_indexing()
    multi_dim_indexing()
    boolean_indexing()
    fancy_indexing()
    advanced_tools()
    indexing_and_views_vs_copies()
    special_indexers()
    examples_summary()