import numpy as np
np.set_printoptions(suppress=True, precision=3)


def arithmetic_and_elementwise():
    x = np.array([1, 2, 3])
    y = np.array([10, 20, 30])
    print("elementwise add, sub, mul, div:")
    print(x + y, x - y, x * y, y / x, sep="\n", end="\n\n")

    # broadcasting example
    M = np.array([[1, 2, 3],
                  [4, 5, 6]])
    v = np.array([10, 20, 30])
    print("broadcasting M + v:\n", M + v, end="\n\n")


def reductions_and_statistics():
    A = np.arange(1, 13).reshape(3, 4)
    print("A:\n", A, sep="\n")
    print("sum, mean, var, std (global):", A.sum(), A.mean(), A.var(), A.std())
    print("sum over axis=0 (cols):", A.sum(axis=0))
    print("sum over axis=1 (rows):", A.sum(axis=1), end="\n\n")

    # min/max with argmin/argmax
    print("min, max, argmin, argmax:", A.min(), A.max(), A.argmin(), A.argmax(), end="\n\n")


def ufuncs_and_vectorization():
    x = np.linspace(0, 2 * np.pi, 5)
    print("x:", x)
    print("sin(x):", np.sin(x))
    print("square, sqrt:", np.square(x), np.sqrt(x))
    # reduce on ufuncs
    print("add.reduce:", np.add.reduce([1, 2, 3, 4]), end="\n\n")


def nan_and_inf_handling():
    a = np.array([1.0, np.nan, 3.0, np.inf])
    print("a:", a)
    print("isnan, isinf:", np.isnan(a), np.isinf(a))
    print("nan-aware sums:", np.nansum(a), "nanmean:", np.nanmean(a), end="\n\n")

def miscellany():
    # unique, bincount, percentile, sort, argsort
    a = np.array([1, 2, 2, 3, 1, 4])
    print("unique:", np.unique(a))
    print("bincount:", np.bincount(a))
    x = np.array([10, 5, 8])
    print("argsort:", np.argsort(x), "sorted:", np.sort(x))
    print("percentile 50 (median):", np.percentile(x, 50), end="\n\n")





if __name__ == "__main__":
    print("NumPy operations notes - concise examples\n")
    arithmetic_and_elementwise()
    reductions_and_statistics()
    ufuncs_and_vectorization()
    nan_and_inf_handling()
    miscellany()