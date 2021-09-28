import numpy as np

import matplotlib.pyplot as plt
from itertools import product
from utils.gauss_proc import *

np.random.seed(6)
np.set_printoptions(precision=4, sign="+")

params_kernel = {'sigma': 1, 'ell': 3.25}


def f(x):
    return np.sin(np.pi * x / 2)**2 + 1

D = 2

dimX = 5

number_of_functions = 6
x = np.linspace(-4, 4, dimX)
Xs = np.array(list(product(*[x for _ in range(D)])))

Sigma0 = Kmatrix(Xs, Xs)

if D == 3:
    X = np.array([[4, 4, 4], [0, 0, 0]])
    f = [2, 4]
elif D == 2:
    X = np.array([[-2, 0], [0, 0], [2, 0]])
    f = [2, 4, -3]
elif D == 1:
    X = np.array([0, 4])
    f = [2, 4]


Xs =  [-4,  4]

print(np.array(X), np.array(Xs))

A = np.array(X)
B = np.array(Xs)
w = (A - B).reshape(A.shape)
print(w)
print("w.shape", w.shape)
print("np.array(X).shape", np.array(X).shape)

new_mean, new_sigma = gaussian_process(X, f, [Xs], Kmatrix, params=params_kernel)

print(Kmatrix(X, [Xs], params_kernel))
print()
print(w * Kmatrix(X, [Xs], params_kernel) / 3.25**2)

print(der_Kmatrix(X, [Xs], params_kernel))
exit()


new_ys = np.random.multivariate_normal(
    mean=new_mean, cov=new_sigma,
    size=number_of_functions)
print(Xs)
print(new_ys)



exit()
# warm up stage
Nwarmup = 2
y_best = np.inf
yi_array = []
xi_array = []
for i in range(Nwarmup):
    x_i = 2 * np.random.random() - 1
    y_i = f(x_i)

    if y_i <= y_best:
        x_best = x_i
        y_best = y_i
        print(f"{x_i:+.4f}, {y_i:+.4f}, accepted")
    else:
        print(f"{x_i:+.4f}, {y_i:+.4f}")
    xi_array.append(x_i)
    yi_array.append(y_i)
# end - warm up stage

X = xi_array
fv = yi_array

AFun = acq_function([0], X, fv, y_best, Kmatrix, params_kernel)

print(AFun)
exit()
NumXs = 40
Xs = np.linspace(-1, 1, NumXs + 1)
AFun = acq_function(Xs, X, fv, y_best, Kmatrix, params_kernel)



plt.plot(Xs, AFun)
plt.show()