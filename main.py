import numpy as np
import math


def tri_diagonal_matrix_algorithm(n, b, c, d, e):
    _x = np.zeros(n)
    for i in range(1, n):
        d[i] = d[i]-c[i]*(e[i-1]/d[i-1])
        b[i] = b[i] - c[i] * (b[i - 1] / d[i - 1])
    _x[n-1] = b[n-1]/d[n-1]
    for i in range(2, n+1):
        _x[n-i] = (b[n-i]-_x[n-i+1]*e[n-i])/d[n-i]
    return _x


def f(_x):
    return math.sin(_x)*math.sin(_x)


def k(_x):
    return math.cos(_x)*math.cos(_x)+1


def k_derivative(_x):
    return -math.sin(2*_x)


x_min = 0
x_max = 1
h1 = 0.1
h2 = 0.01
n1 = int((x_max-x_min)/h1+1)
n2 = int((x_max-x_min)/h2+1)
x_h1 = np.linspace(x_min, x_max, n1)
x_h2 = np.linspace(x_min, x_max, n2)
print("Аппроксимация поставленной задачи разностной схемой второго порядка")
b1_1 = np.zeros(n1)
c1_1 = np.zeros(n1)
d1_1 = np.zeros(n1)
e1_1 = np.zeros(n1)

b1_1[0] = 0
d1_1[0] = 1
e1_1[0] = (-2.0)/(h1*h1+h1+2)

b1_1[n1-1] = 1+(h1/2)*f(x_max)+(h1*k_derivative(x_max)/(2*k(x_max)))
c1_1[n1-1] = -k(x_max)/h1
d1_1[n1-1] = 1+k(x_max)/h1+h1/2+h1*k_derivative(x_max)/(2*k(x_max))


for j in range(1, n1-1):
    b1_1[j] = -f(x_h1[j])
    c1_1[j] = -k_derivative(x_h1[j])/(2*h1) + k(x_h1[j])/(h1*h1)
    d1_1[j] = -2 * k(x_h1[j])/(h1*h1) - 1
    e1_1[j] = k_derivative(x_h1[j])/(2*h1)+k(x_h1[j])/(h1*h1)

y = tri_diagonal_matrix_algorithm(n1, b1_1, c1_1, d1_1, e1_1)
print("Y=", y)
