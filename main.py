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
    return math.sin(_x)**2


def k(_x):
    return math.cos(_x)**2+1


def k_derivative(_x):
    return -math.sin(2*_x)


def approx(_h, _n, _x):
    print("Аппроксимация поставленной задачи разностной схемой второго порядка")
    b1_1 = np.zeros(_n)
    c1_1 = np.zeros(_n)
    d1_1 = np.zeros(_n)
    e1_1 = np.zeros(_n)

    b1_1[0] = 0
    d1_1[0] = 1
    e1_1[0] = (-2.0)/(_h * _h + _h + 2)

    b1_1[_n - 1] = 1 + (_h / 2) * f(x_max) + (_h * k_derivative(x_max) / (2 * k(x_max)))
    c1_1[_n - 1] = -k(x_max) / _h
    d1_1[_n - 1] = 1 + k(x_max) / _h + _h / 2 + _h * k_derivative(x_max) / (2 * k(x_max))

    for j in range(1, _n - 1):
        b1_1[j] = -f(_x[j])
        c1_1[j] = -k_derivative(_x[j]) / (2 * _h) + k(_x[j]) / (_h * _h)
        d1_1[j] = -2 * k(_x[j]) / (_h * _h) - 1
        e1_1[j] = k_derivative(_x[j]) / (2 * _h) + k(_x[j]) / (_h * _h)
    y1_1 = tri_diagonal_matrix_algorithm(_n, b1_1, c1_1, d1_1, e1_1)
    print("Решение для h=", _h, " Y=", y1_1)


def iim(_h, _n, _x):
    print("Интегро-интерполяционный метод")
    a_iim = np.zeros(_n - 1)
    d_iim = np.zeros(_n)
    fi_iim = np.zeros(_n)
    b2_1 = np.zeros(_n)
    c2_1 = np.zeros(_n)
    d2_1 = np.zeros(_n)
    e2_1 = np.zeros(_n)

    d_iim[0] = 1
    d_iim[_n - 1] = 1
    fi_iim[0] = f(0.25 * _h)
    fi_iim[_n - 1] = f(1 - 0.25 * _h)

    for j in range(0, _n - 1):
        a_iim[j] = k((_x[j]+_x[j+1])/2)
    for j in range(1, _n - 1):
        d_iim[j] = 1
        fi_iim[j] = f(_x[j])

    b2_1[0] = _h * fi_iim[0] / 2
    d2_1[0] = 1 + _h * d_iim[0] / 2 + a_iim[0] / _h
    e2_1[0] = - a_iim[0] / _h

    b2_1[_n - 1] = 1 + _h * fi_iim[_n - 1] / 2
    c2_1[_n - 1] = - a_iim[_n - 2] / _h
    d2_1[_n - 1] = 1 + _h * d_iim[_n - 1] / 2 + a_iim[_n - 2] / _h

    for j in range(1, _n - 1):
        b2_1[j] = -fi_iim[j]
        c2_1[j] = a_iim[j-1]/(_h * _h)
        d2_1[j] = -(a_iim[j]+a_iim[j-1]) / (_h * _h) - d_iim[j]
        e2_1[j] = a_iim[j]/(_h * _h)

    y2_1 = tri_diagonal_matrix_algorithm(_n, b2_1, c2_1, d2_1, e2_1)
    print("Решение для h=", _h, " Y=", y2_1)


def vdm(_h, _n, _x):
    print("Вариационно-разностный метод")
    a_vdm = np.zeros(_n - 1)
    d_vdm = np.zeros(_n)
    fi_vdm = np.zeros(_n)
    b3_1 = np.zeros(_n)
    c3_1 = np.zeros(_n)
    d3_1 = np.zeros(_n)
    e3_1 = np.zeros(_n)

    d_vdm[0] = 1
    d_vdm[_n - 1] = 1
    fi_vdm[0] = f(0)
    fi_vdm[_n - 1] = f(1)

    for j in range(0, _n - 1):
        a_vdm[j] = (k(_x[j])+k(_x[j+1]))/2
    for j in range(1, _n - 1):
        d_vdm[j] = 1
        fi_vdm[j] = (f(_x[j])*(_x[j+1]-_x[j])+f(_x[j])*(_x[j]-_x[j-1]))/(2 * _h)

    b3_1[0] = _h * fi_vdm[0] / 2
    d3_1[0] = 1 + _h * d_vdm[0] / 2 + a_vdm[0] / _h
    e3_1[0] = - a_vdm[0] / _h

    b3_1[_n - 1] = 1 + _h * fi_vdm[_n - 1] / 2
    c3_1[_n - 1] = - a_vdm[_n - 2] / _h
    d3_1[_n - 1] = 1 + _h * d_vdm[_n - 1] / 2 + a_vdm[_n - 2] / _h

    for j in range(1, _n - 1):
        b3_1[j] = -fi_vdm[j]
        c3_1[j] = a_vdm[j-1]/(_h * _h)
        d3_1[j] = -(a_vdm[j]+a_vdm[j-1]) / (_h * _h) - d_vdm[j]
        e3_1[j] = a_vdm[j]/(_h * _h)

    y3_1 = tri_diagonal_matrix_algorithm(_n, b3_1, c3_1, d3_1, e3_1)
    print("Решение для h=", _h, " Y=", y3_1)


x_min = 0
x_max = 1
h1 = 0.1
h2 = 0.01
n1 = int((x_max - x_min) / h1 + 1)
n2 = int((x_max-x_min)/h2+1)
x_h1 = np.linspace(x_min, x_max, n1)
x_h2 = np.linspace(x_min, x_max, n2)
approx(h1, n1, x_h1)
iim(h1, n1, x_h1)
vdm(h1, n1, x_h1)

approx(h2, n2, x_h2)
iim(h2, n2, x_h2)
vdm(h2, n2, x_h2)
