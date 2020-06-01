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
print("b1_1:", b1_1)
print("c1_1:", c1_1)
print("d1_1:", d1_1)
print("e1_1:", e1_1)
y1_1 = tri_diagonal_matrix_algorithm(n1, b1_1, c1_1, d1_1, e1_1)
print("Решение для h=0.1 Y=", y1_1)


print("Интегро-интерполяционный метод")
a_iim = np.zeros(n1-1)
d_iim = np.zeros(n1)
fi_iim = np.zeros(n1)
b2_1 = np.zeros(n1)
c2_1 = np.zeros(n1)
d2_1 = np.zeros(n1)
e2_1 = np.zeros(n1)

d_iim[0] = 1
d_iim[n1-1] = 1
fi_iim[0] = f(0.25*h1)
print("fi[0]=", fi_iim[0])
fi_iim[n1-1] = f(1 - 0.25*h1)

for j in range(0, n1-1):
    a_iim[j] = k((x_h1[j]+x_h1[j+1])/2)
for j in range(1, n1 - 1):
    d_iim[j] = 1
    fi_iim[j] = f(x_h1[j])

b2_1[0] = h1*fi_iim[0]/2
d2_1[0] = 1 + h1*d_iim[0]/2 + a_iim[0]/h1
e2_1[0] = - a_iim[0]/h1

b2_1[n1-1] = 1+h1*fi_iim[n1-1]/2
c2_1[n1-1] = - a_iim[0]/h1
d2_1[n1-1] = 1 + h1*d_iim[n1-1]/2 + a_iim[n1-2]/h1

for j in range(1, n1-1):
    b2_1[j] = -f(x_h1[j])
    c2_1[j] = a_iim[j-1]/(h1*h1)
    d2_1[j] = -(a_iim[j]+a_iim[j-1])/(h1*h1)-d_iim[j]
    e2_1[j] = a_iim[j]/(h1*h1)

print("b2_1:", b2_1)
print("c2_1:", c2_1)
print("d2_1:", d2_1)
print("e2_1:", e2_1)

y2_1 = tri_diagonal_matrix_algorithm(n1, b2_1, c2_1, d2_1, e2_1)
print("Решение для h=0.1 Y=", y2_1)

print("Вариационно-разностный метод")
a_vdm = np.zeros(n1-1)
d_vdm = np.zeros(n1)
fi_vdm = np.zeros(n1)
b3_1 = np.zeros(n1)
c3_1 = np.zeros(n1)
d3_1 = np.zeros(n1)
e3_1 = np.zeros(n1)

d_vdm[0] = 1
d_vdm[n1-1] = 1
fi_vdm[0] = f(0)
fi_vdm[n1-1] = f(1)

for j in range(0, n1-1):
    a_vdm[j] = (k(x_h1[j])+k(x_h1[j+1]))/2
for j in range(1, n1 - 1):
    d_vdm[j] = (x_h1[j+1]-x_h1[j-1])/2*h1
    fi_iim[j] = (f(x_h1[j])*(x_h1[j+1]-x_h1[j])+f(x_h1[j])*(x_h1[j]-x_h1[j-1]))/2*h1

b3_1[0] = h1*fi_vdm[0]/2
d3_1[0] = 1 + h1*d_vdm[0]/2 + a_vdm[0]/h1
e3_1[0] = - a_vdm[0]/h1

b3_1[n1-1] = 1+h1*fi_vdm[n1-1]/2
c3_1[n1-1] = - a_vdm[0]/h1
d3_1[n1-1] = 1 + h1*d_vdm[n1-1]/2 + a_vdm[n1-2]/h1

for j in range(1, n1-1):
    b3_1[j] = -f(x_h1[j])
    c3_1[j] = a_vdm[j-1]/(h1*h1)
    d3_1[j] = -(a_vdm[j]+a_vdm[j-1])/(h1*h1)-d_vdm[j]
    e3_1[j] = a_vdm[j]/(h1*h1)

print("b3_1:", b3_1)
print("c3_1:", c3_1)
print("d3_1:", d3_1)
print("e3_1:", e3_1)

y3_1 = tri_diagonal_matrix_algorithm(n1, b3_1, c3_1, d3_1, e3_1)
print("Решение для h=0.1 Y=", y3_1)
