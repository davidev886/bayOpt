import numpy as np
from itertools import product
from numpy.linalg import norm
import scipy

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import seaborn as sns



np.random.seed(6)




def Kmatrix(A, B, sigma=1, ell=1):
    """Exponentiated quadratic kernel"""
    A = np.array(A)
    B = np.array(B)
    kernel_matrix = [[sigma**2 * np.exp(-norm(xa - xb)**2 / (2 * ell**2)) for xa in A] for xb in B]
    return kernel_matrix


def gaussian_process(X, f, Xs, Kmatrix):
    K_XX = Kmatrix(X, X)
    K_XXs = Kmatrix(Xs, X)
    K_XsXs = Kmatrix(Xs, Xs)

    solved_kernel = scipy.linalg.solve(K_XX, K_XXs, assume_a='pos')
    new_mean = solved_kernel.T @ f
    new_sigma_temp = K_XsXs - solved_kernel.T @ K_XXs
    new_sigma = (new_sigma_temp + new_sigma_temp.T) / 2.
    return new_mean, new_sigma


def f(x):
    return np.sin(np.pi * x)**2 + 1

dimX = 51
D = 1


Nwarmup = 4
N_bayes_opt = 10
y_best = np.inf

yi_array = []
xi_array = []
y_best_array = []
x_best_array = []

np.set_printoptions(precision=4, sign="+")

for i in range(Nwarmup):
    x_i = 2 * np.random.random() - 1

    y_i = f(x_i)

    if y_i <= y_best:
        x_best = x_i
        y_best = y_i
        y_best_array.append(y_i)
        x_best_array.append(x_i)
        print(f"{x_i:+.4f}, {y_i:+.4f}, accepted")
    else:
        print(f"{x_i:+.4f}, {y_i:+.4f}")
    xi_array.append(x_i)
    yi_array.append(y_i)


# for i in range(N_bayes_opt):
#     Xs = xi_array
#     k_matrix = Kmatrix(xi_array, xi_array)
#     new_mean, new_sigma = gaussian_process(X, yi_array, Xs, Kmatrix)

print()
print(np.array(x_best_array))
print(np.array(y_best_array))

exit()
number_of_functions = 6
x = np.linspace(-4, 4, dimX)
Xs = np.array(list(product(*[x for _ in range(D)])))

Sigma0 = Kmatrix(Xs, Xs)

if D == 3:
    X = np.array([[4, 4, 4], [0, 0, 0]])
elif D == 2:
    X = np.array([[-2, 0], [0,0]])
elif D == 1:
    X = np.array([0, 4])

f = [2, 4]

new_mean, new_sigma = gaussian_process(X, f, Xs, Kmatrix)

new_ys = np.random.multivariate_normal(
    mean=new_mean, cov=new_sigma,
    size=number_of_functions)
print(Xs)
print(new_ys)



if D == 1:
    plt.figure(figsize=(6, 4))
    # for i in range(number_of_functions):
    #     plt.plot(Xs, ys[i], linestyle='-', marker='o', markersize=3)

    for i in range(number_of_functions):
        plt.plot(Xs, new_ys[i], linestyle='-', marker='o', markersize=3)



    plt.xlim([-4, 4])
    plt.show()