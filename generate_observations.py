import numpy as np
from matplotlib import pyplot as plt
from rk4 import rk4


def perfect_model(t, x):
    """
    Returns dx/dt for the Lorentz63 attractor
    :param t: time
    :param x: position
    :return: dx/dt is expressed as a numpy vector
    """
    # Define the parameters of the Lorentz63 attractor
    sigma = 10
    rho = 28
    beta = 8 / 3
    return np.array([sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - beta * x[2]])


# Introduce different time_step and integration step size????
def generate_observation(initial_x, initial_t, final_t, step, std):
    """
    Solves the Lorentz63 ODE and adds noise (Gaussian error) to the solution
    :param initial_x: initial position
    :param initial_t: initial time
    :param final_t: final time
    :param step: Fixed time step for integration
    :param std: standard deviation of the Gaussian error
    :return:
    """
    num = int((final_t - initial_t) / step)

    t, x = rk4(initial_x, initial_t, perfect_model, step, num)

    # standard deviation of the Gaussian error added
    s = x + np.random.normal(0, std, (len(initial_x), num + 1))

    return t, s


def predict_imperfect(initial_x, initial_t, c, time_step, integration_time_step):
    """
    Solves the ODE dx/dt = f(t,x) with f given by the function derivative below
    :param initial_x: Initial position
    :param initial_t: Initial time
    :param c: Controls the inaccuracy of the model
    :param time_step: Fixed time step
    :param integration_time_step: Time step used for the Runge-Kutta integration
    :return: A numpy vector of position at time = initial_t + time_step
    """

    def derivative(t, x):
        # We replace x with c * np.sin(x/c) in the differential equations
        return perfect_model(t, c * np.sin(x / c))

    num = int(time_step / integration_time_step)
    t, x = rk4(initial_x, initial_t, derivative, integration_time_step, num)

    return x[:, -1]


initial_x = np.array([3, 8, 5])
initial_t = 0
final_t = 10
c = 100
time_step = 0.01
integration_time_step = 0.01
std = 0

def difference_predict(initial_x, initial_t, final_t, c, time_step, integration_time_step, std):
    num = int((final_t - initial_t)/ time_step)
    t, s = generate_observation(initial_x, initial_t, final_t, time_step, std)

    # Plot the observations 3D
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(s[0, :], s[1, :], s[2, :], linewidth=0.2)
    plt.show()

    difference = np.zeros((len(initial_x), num + 1))
    predictions = np.zeros((len(initial_x), num + 1))
    predictions[:,0] = initial_x
    for i in range(len(t[:-1])):
        predictions[:, i + 1] = predict_imperfect(s[:,i], t[i], c, time_step, integration_time_step)
        difference[:, i + 1] = predictions[:, i + 1] - s[:, i+1]
    return difference


delta = difference_predict(initial_x, initial_t, final_t, c, time_step, integration_time_step, std)

print(delta.shape)
print(delta)
plt.scatter(delta[0, :], delta[1, :])
plt.show()
plt.scatter(delta[0, :], delta[2, :])
plt.show()
ax = plt.figure().add_subplot(projection='3d')
ax.plot(delta[0,:], delta[1,:], delta[2,:], linewidth=0.2)
plt.show()
