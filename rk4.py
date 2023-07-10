import numpy as np
from typing import Callable


def rk4(initial_x: np.ndarray, initial_t: float, f: Callable, step: float, num: int):
    """
    Solves the ODE dx/dt = f(t, x) by 4th order Runge-Kutta
    :param initial_x: Initial x value
    :param initial_t: Initial t value
    :param f: Function such that dx/dt = f(t, x)
    :param step: Size of fixed time step
    :param num: Number of time steps
    :return: Tuple of vectors; time and position of the solution
    """
    t = np.arange(0, num) * step + initial_t
    x = np.zeros((len(initial_x), num))
    x[:, 0] = initial_x
    for i, time in enumerate(t[:-1]):
        k1 = f(time, x[:, i])
        k2 = f(time + step / 2, x[:, i] + step * k1 / 2)
        k3 = f(time + step / 2, x[:, i] + step * k2 / 2)
        k4 = f(time + step, x[:, i] + step * k3)
        x[:, i + 1] = x[:, i] + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return t, x
