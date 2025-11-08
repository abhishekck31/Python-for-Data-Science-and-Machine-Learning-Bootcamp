import numpy as np

# Introduction to NumPy
"""
NumPy: definition and common applications

Definition:
NumPy (Numerical Python) is a core library for numerical computing in Python.
It provides the ndarray: a fast, homogeneous, N-dimensional array object,
together with optimized routines for elementwise operations, broadcasting,
linear algebra, random number generation, FFTs, and more.

Common applications:
- Scientific computing and simulations
- Data preprocessing and numerical feature engineering
- Linear algebra, matrix computations, and solving systems of equations
- Statistics, probability, and descriptive analytics
- Machine learning (base array operations for libraries like scikit-learn, TensorFlow, PyTorch)
- Signal processing, image processing, and computer vision
- Financial modeling and quantitative analysis

This small demo shows basic array creation and operations.
"""


def numpy_intro_demo():
    a = np.array([1.0, 2.0, 3.0])         # 1D array
    b = np.eye(3)                         # 3x3 identity matrix
    c = a * 2                             # elementwise scaling
    mean_a = a.mean()                     # aggregate operation
    matvec = b @ a                        # matrix-vector product

    print("a =", a)
    print("c = a * 2 =", c)
    print("mean(a) =", mean_a)
    print("b (identity 3x3):\n", b)
    print("b @ a =", matvec)

if __name__ == "__main__":
    numpy_intro_demo()