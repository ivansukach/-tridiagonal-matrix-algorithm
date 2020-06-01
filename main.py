import numpy as np


def tri_diagonal_matrix_algorithm(n, b, c, d, e):
    _x = np.zeros(n)
    for i in range(1, n):
        d[i] = d[i]-c[i]*(e[i-1]/d[i-1])
        b[i] = b[i] - c[i] * (b[i - 1] / d[i - 1])
    _x[n-1] = b[n-1]/d[n-1]
    for i in range(2, n+1):
        _x[n-i] = (b[n-i]-_x[n-i+1]*e[n-i])/d[n-i]
    return _x


x = tri_diagonal_matrix_algorithm(5, [4, 22, 22, 17, 2], [0, 3, 3, 1, 3], [2, 2, 4, 1, -2], [1, 5, 1, 2, 0])
print("X=", x)
