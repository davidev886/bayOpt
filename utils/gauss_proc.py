import numpy as np
from numpy.linalg import norm
import scipy
from scipy.stats import norm as gaussian_normal


def Kmatrix(A, B, sigma=1, ell=1):
    """Exponentiated quadratic kernel:
        returns a list that has len(A) rows and len(B) columns
    """

    A = np.array(A)
    B = np.array(B)

    kernel_matrix = [[sigma**2 * np.exp(-norm(xa - xb)**2 / (2 * ell**2))
                     for xb in B]
                     for xa in A]
    return kernel_matrix


def gaussian_process(X, f, Xs, Kmatrix, params={'sigma': 1, 'ell': 1}):
    sigma = params['sigma']
    ell = params['ell']
    K_XX = Kmatrix(X, X, sigma, ell)
    K_XXs = Kmatrix(X, Xs, sigma, ell)
    K_XsXs = Kmatrix(Xs, Xs, sigma, ell)
    # solve for w in the system K(X,X) w = K(X, Xs)
    # then consider w^T = K(Xs, X).K(X, X)^-1
    # for computing the new mean and new variance
    solved_w = scipy.linalg.solve(K_XX, K_XXs, assume_a='pos')
    new_mean = solved_w.T @ f
    new_sigma_temp = K_XsXs - solved_w.T @ K_XXs
    new_sigma = (new_sigma_temp + new_sigma_temp.T) / 2.
    return new_mean, new_sigma


def acq_function(Xs, X, fv, y_best, Kmatrix, params_kernel):
    new_mean, new_sigma = gaussian_process(X, fv, [Xs], Kmatrix, params_kernel)

    cdf = gaussian_normal.cdf(x=(y_best - new_mean[0])/new_sigma[0, 0])
    pdf = gaussian_normal.pdf(x=(y_best - new_mean[0])/new_sigma[0, 0])

    AFun = new_sigma[0, 0] * (pdf +
                              (y_best - new_mean[0])/new_sigma[0, 0] * cdf)
    return AFun


def acq_function_optimize(Xs, X, fv, y_best, Kmatrix, params_kernel):
    new_mean, new_sigma = gaussian_process(X, fv, [Xs], Kmatrix, params_kernel)

    cdf = gaussian_normal.cdf(x=(y_best - new_mean[0])/new_sigma[0, 0])
    pdf = gaussian_normal.pdf(x=(y_best - new_mean[0])/new_sigma[0, 0])

    AFun = new_sigma[0, 0] * (pdf +
                              (y_best - new_mean[0])/new_sigma[0, 0] * cdf)
    return -AFun
