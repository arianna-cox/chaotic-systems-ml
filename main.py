import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from rk4 import rk4

sigma = 10
rho = 28
beta = 8 / 3


def derivative(t, x):
    return np.array([sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - beta * x[2]])


t_span = np.array([0, 100])
x0 = np.array([6, 4, 1])

def test_f(t, x):
    return np.array([0,0,0])

# result1 = solve_ivp(derivative, t_span, x0, max_step=0.001)
t, result2 = rk4(x0, 0, f=test_f, num=int(100 / 0.01), step=0.001)
# print(result1.y[0])
# print(len(result1.y[0]))
# print(result1.t)

print(result2[0,:])
print(len(result2[0,:]))
print(t)

# ax = plt.figure().add_subplot(projection='3d')
# ax.plot(result.y[0], result.y[1], result.y[2], linewidth=0.1)
# plt.show()
