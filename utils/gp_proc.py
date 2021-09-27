import numpy as np
from numpy.linalg import norm
import scipy
from scipy.stats import norm as gaussian_normal


def Kmatrix(A, B, sigma=1, ell=1):
    """Exponentiated quadratic kernel"""
    A = np.array(A)
    B = np.array(B)

    kernel_matrix = [[sigma**2 * np.exp(-norm(xa - xb)**2 / (2 * ell**2))
                     for xa in A]
                     for xb in B]
    return kernel_matrix


def gaussian_process(X, f, Xs, Kmatrix, params={'sigma': 1, 'ell': 1}):
    sigma = params['sigma']
    ell = params['ell']
    K_XX = Kmatrix(X, X, sigma, ell)
    K_XXs = Kmatrix(Xs, X, sigma, ell)
    K_XsXs = Kmatrix(Xs, Xs, sigma, ell)

    solved_kernel = scipy.linalg.solve(K_XX, K_XXs, assume_a='pos')
    new_mean = solved_kernel.T @ f
    new_sigma_temp = K_XsXs - solved_kernel.T @ K_XXs
    new_sigma = (new_sigma_temp + new_sigma_temp.T) / 2.
    return new_mean, new_sigma


def acq_function(Xs, X, fv, y_best, Kmatrix, params_kernel):
    Xs = np.array(Xs)
    if Xs.shape == ():
        Xs = np.array([Xs])

    AFun = []
    for xs in Xs:
        new_mean, new_sigma = gaussian_process(X, fv, [xs], Kmatrix, params_kernel)

        cdf = gaussian_normal.cdf(x = (y_best - new_mean[0])/new_sigma[0, 0])
        pdf = gaussian_normal.pdf(x = (y_best - new_mean[0])/new_sigma[0, 0])

        AFun.append(new_sigma[0, 0] * (pdf + (y_best - new_mean[0])/new_sigma[0, 0] * cdf))

    return AFun