import numpy as np
from numpy.linalg import norm
import scipy
from scipy.stats import norm as gaussian_normal


def Kmatrix_gaussian(A, B, params={'sigma': 1, 'ell': 1}):
    """Exponentiated quadratic kernel:
        returns a list that has len(A) rows and len(B) columns
    """
    sigma = params['sigma']
    ell = params['ell']
    A = np.array(A)
    B = np.array(B)

    kernel_matrix = [[sigma**2 * np.exp(-norm(xa - xb)**2 / (2 * ell**2))
                     for xb in B]
                     for xa in A]

    return np.array(kernel_matrix)


def der_Kmatrix(A, B, params={'sigma': 1, 'ell': 1}):
    """Derivative of the exponentiated quadratic kernel:
       returns the derivative of the kernel wrt the matrix B
    """

    K_AB = Kmatrix_gaussian(A, B, params)
    ell = params['ell']
    A = np.array(A)
    B = np.array(B)
    w = (A - B).reshape(A.shape)

    derivative_matrix = w * K_AB / ell**2

    return derivative_matrix


class gaussian_process(object):
    def __init__(self,
                 X,
                 f,
                 Kmatrix=Kmatrix_gaussian,
                 params={'sigma': 1, 'ell': 1}
                 ):
        self.X = X
        self.f = f
        self.Kmatrix = Kmatrix
        self.K_XX = self.Kmatrix(X, X, params)
        self.params = params
        return None

    def predict(self, Xs):
        K_XXs = self.Kmatrix(self.X, Xs, self.params)
        K_XsXs = self.Kmatrix(Xs, Xs, self.params)

        # solve for w in the system K(X,X) w = K(X, Xs)
        # then consider w^T = K(Xs, X).K(X, X)^-1
        # for computing the new mean and new variance
        solved_w = scipy.linalg.solve(self.K_XX, K_XXs, assume_a='pos')
        new_mean = solved_w.T @ self.f
        new_sigma_temp = K_XsXs - solved_w.T @ K_XXs
        new_sigma = (new_sigma_temp + new_sigma_temp.T) / 2.
        return new_mean, new_sigma

    def der_mean(self, Xs):
        DK_XXs = der_Kmatrix(self.X, Xs, self.params)

        # solve for w in the system K(X,X) w = K(X, Xs)
        # then consider w^T = K(Xs, X).K(X, X)^-1
        # for computing the new mean and new variance
        solved_w = scipy.linalg.solve(self.K_XX, DK_XXs, assume_a='pos')
        der_mean = solved_w.T @ self.f

        return der_mean

    def acq_function(self, Xs, fv, y_best):
        new_mean, new_sigma = self.predict([Xs])

        cdf = gaussian_normal.cdf(x=(y_best - new_mean[0])/new_sigma[0, 0])
        pdf = gaussian_normal.pdf(x=(y_best - new_mean[0])/new_sigma[0, 0])

        AFun = new_sigma[0, 0] * (pdf +
                                  (y_best - new_mean[0])/new_sigma[0, 0] * cdf)
        return AFun

    def acq_function_optimize(self, Xs, fv, y_best):
        new_mean, new_sigma = self.predict([Xs])

        cdf = gaussian_normal.cdf(x=(y_best - new_mean[0])/new_sigma[0, 0])
        pdf = gaussian_normal.pdf(x=(y_best - new_mean[0])/new_sigma[0, 0])

        AFun = new_sigma[0, 0] * (pdf +
                                  (y_best - new_mean[0])/new_sigma[0, 0] * cdf)
        return -AFun
