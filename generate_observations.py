import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

# Define the perfect model of the Lorentz63 attractor
sigma = 10
rho = 28
beta = 8 / 3


def perfect_model(t, x):
    return [sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - beta * x[2]]


def generate_observation(time_step, final_time, std, integration_time_step, x0):
    length = int(final_time / time_step) + 1

    result = solve_ivp(perfect_model, [0, final_time], x0, min_step=integration_time_step,
                       max_step=integration_time_step)

    # standard deviation of the Gaussian error added
    s = np.array([result.y[i][::10] for i in range(3)]) + np.random.normal(0, std, (3, length))

    return s


final_time = 100
std = 0
s = generate_observation(0.1, final_time, std, 0.01, [3, 8, 5])


# ax = plt.figure().add_subplot(projection='3d')
# ax.plot(s[0], s[1], s[2], linewidth=0.2)
# plt.show()

def F(x, c, time_step, integration_time_step):
    # x is modified in the imperfect function f
    # We replace x with c * np.sin(x/c) in the differential equations
    def derivative(t, x):
        return perfect_model(t, c * np.sin(x/c))

    result = solve_ivp(derivative, [0, time_step], x, min_step=integration_time_step, max_step=integration_time_step)
    # PROBLEM!!!
    print(result.t)
    print(result.y[0])
    return [result.y[0][-1], result.y[1][-1], result.y[2][-1]]


delta = np.zeros((3, len(s[0]) - 1))
Fs = np.zeros((3, len(s[0]) - 1))
c = 100
for i in range(len(s[0]) - 1):
    Fs[:, i] = F(s[:, i], c, 0.1, 0.01)
    delta[:, i] = s[:, i + 1] - Fs[:, i]
print('delta')
plt.scatter(delta[0, :], delta[1, :])
plt.show()
plt.scatter(delta[0, :], delta[2, :])
plt.show()
ax = plt.figure().add_subplot(projection='3d')
ax.plot(Fs[0], Fs[1], Fs[2], linewidth=0.2)
plt.show()

from rk4 import rk4

