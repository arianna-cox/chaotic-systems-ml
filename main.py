import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from rk4 import rk4

sigma = 10
rho = 28
beta = 8 / 3


def derivative(t, x):
    return np.array([sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - beta * x[2]])

