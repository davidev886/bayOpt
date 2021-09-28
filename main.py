import numpy as np
from itertools import product
import matplotlib.pyplot as plt

from scipy.optimize import minimize


from utils.gauss_proc import *
from utils.qaoa import *

np.random.seed(6)
np.set_printoptions(precision=4, sign="+", suppress=True)

params_kernel = {'sigma': 1, 'ell': 1000}

gamma_range = [1000, 2000]   # extremes where to search for the values of gamma and beta
beta_range = [500, 1000]

num_level_p = 1

dimX = 11# number of test points for computing the acquisition function

Nwarmup = 3

pos = np.array([[0., 0.],
                [-4, -7],
                [4, -7],
                [8, 6],
                [-8, 6]]
               )

def plot_acquisition_function(gamma_range, beta_range, dimX, num_level_p):
    gamma_arr = np.linspace(gamma_range[0], gamma_range[1], dimX)
    beta_arr = np.linspace(beta_range[0], beta_range[1], dimX)
    Xs = np.array(list(product(*[gamma_arr, beta_arr] * num_level_p)))
    AFun = []
    for xs in Xs:
        AFun.append(acq_function(xs, X_train, y_train, y_best, Kmatrix, params_kernel))
    AFun = np.array(AFun)

    fig = plt.figure()
    plt.imshow(AFun.reshape((dimX, dimX)).T,
                extent=[min(gamma_arr),max(gamma_arr),min(beta_arr),max(beta_arr)],
                origin="lower")

    plt.xlabel("gamma")
    plt.ylabel("beta")


    plt.colorbar()
    plt.show()


# Grafo di pasqal originale
# pos = np.array([[0,0], [0,10],[0,-10],[10,0],[-10,0]])

G = pos_to_graph(pos)

qubits = dict(enumerate(pos))
reg = Register(qubits)

# warm up stage
y_best = np.inf
y_train = []
X_train = []
for i in range(Nwarmup):
    gamma_random = np.random.randint(gamma_range[0],
                                     gamma_range[1],
                                     size=num_level_p
                                     )
    beta_random = np.random.randint(beta_range[0],
                                    beta_range[1],
                                    size=num_level_p)

    X = np.array([gamma_random, beta_random]).T.ravel()
    Y = apply_qaoa(X, reg, G)

    if Y <= y_best:
        x_best = X
        y_best = Y
        print(f"{X}, {Y:+.4f}, accepted")
    else:
        print(f"{X}, {Y:+.4f}, not accepted")

    X_train.append(X)
    y_train.append(Y)

print(y_best)

# end - warm up stage


if 1:
    gamma_random = np.random.randint(gamma_range[0],
                                     gamma_range[1],
                                     size=num_level_p
                                     )
    beta_random = np.random.randint(beta_range[0],
                                    beta_range[1],
                                    size=num_level_p)

    x0 = np.array([gamma_random, beta_random]).T.ravel()


    res = minimize(acq_function_optimize,
                   x0,
                   args=(X_train, y_train, y_best, Kmatrix, params_kernel),
                   method='Nelder-Mead',
                   options={'disp': True},
                   tol=1e-6)

    print(res)

if num_level_p == 1:
    plot_acquisition_function(gamma_range, beta_range, dimX, num_level_p)


exit()



plt.plot(Xs, AFun)
plt.show()


N_train = 10
N_test = 50  # Number of test elements
iterations = 1
gamma_range = [1000,10000]   # extremes where to search for the values of gamma and beta
beta_range = [1000,10000]

method = 'FD'

#------- TRAIN ON N DATA --------#
X_train = []   #data
y_train = []   #label

for i in range(N_train):
    X = [np.random.randint(gamma_range[0],gamma_range[1]), np.random.randint(beta_range[0],beta_range[1])]
    X_train.append(X)
    Y = apply_qaoa(X)
    y_train.append(Y)