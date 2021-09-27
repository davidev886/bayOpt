import numpy as np

import matplotlib.pyplot as plt

from utils.gp_proc import *

np.random.seed(6)
np.set_printoptions(precision=4, sign="+")

params_kernel = {'sigma': 1, 'ell': 0.25}


def f(x):
    return np.sin(np.pi * x / 2)**2 + 1

D = 1
N_bayes_opt = 10

# warm up stage
Nwarmup = 4
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
NumXs = 40
Xs = np.linspace(-1, 1, NumXs + 1)
AFun = acq_function(Xs, X, fv, y_best, Kmatrix, params_kernel)



plt.plot(Xs, AFun)
plt.show()
