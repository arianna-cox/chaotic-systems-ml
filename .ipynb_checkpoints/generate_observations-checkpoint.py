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


def generate_observation(initial_x, initial_t, final_t, step, integration_step, std):
    """
    Solves the Lorentz63 ODE and adds noise (Gaussian error) to the solution
    :param initial_x: initial position
    :param initial_t: initial time
    :param final_t: final time
    :param step: Fixed time step for integration
    :param std: standard deviation of the Gaussian error
    :return: numpy arrays of time and observations
    """

    num_integration = int((final_t - initial_t) / integration_step)
    t_full, x_full = rk4(initial_x, initial_t, perfect_model, integration_step, num_integration)

    t, x = t_full[0::int(step/integration_step)], x_full[0::int(step/integration_step)]

    # standard deviation of the Gaussian error added
    num = int((final_t - initial_t) / step)
    s = x + np.random.normal(0, std, (num + 1, len(initial_x)))

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

    return x[-1, :]


def predict(initial_x, initial_t, final_t, c, time_step, integration_time_step, number_timesteps_predict, std):
    num = int((final_t - initial_t) / time_step)
    t, s = generate_observation(initial_x, initial_t, final_t, time_step, integration_time_step, std)

    # # Plot the observations 3D
    # ax = plt.figure().add_subplot(projection='3d')
    # ax.plot(s[:, 0], s[:, 1], s[:, 2], linewidth=0.2)
    # plt.show()

    predictions = np.zeros((num + 1, len(initial_x)))
    predictions[:number_timesteps_predict, :] = s[:number_timesteps_predict,:]
    for i in range(len(t[:-number_timesteps_predict])):
        predictions[i + number_timesteps_predict, :] = predict_imperfect(s[i, :], t[i], c, time_step * number_timesteps_predict, integration_time_step)
    return t, s, predictions


def generate_data(number_of_samples, time_span, c, time_step, integration_time_step, number_timesteps_predict, std):
    all_s = np.zeros((number_of_samples, int(time_span / time_step) + 1, 3))
    all_predictions = np.zeros((number_of_samples, int(time_span / time_step) + 1, 3))
    for i in range(number_of_samples):
        # Randomly generate initial value for x
        x0 = 10 * np.random.rand(3) + 2
        t, all_s[i], all_predictions[i] = predict(x0, 0, time_span, c, time_step, integration_time_step, number_timesteps_predict, std)
        print(i)
    return t, all_s, all_predictions


if __name__ == "__main__":
    initial_x = np.array([3, 8, 5])
    initial_t = 0
    final_t = 100
    c = 100
    time_step = 0.01
    integration_time_step = 0.01
    std = 0

    t, s, prediction = predict(initial_x, initial_t, final_t, c, time_step, integration_time_step, std)
    delta = prediction - s

    print(delta.shape)
    print(delta)
    plt.scatter(delta[:, 0], delta[:, 1], s=5)
    plt.ylabel('delta_y')
    plt.xlabel('delta_x')
    plt.show()
    plt.scatter(delta[:, 0], delta[:, 2], s=5)
    plt.ylabel('delta_z')
    plt.xlabel('delta_x')
    plt.show()
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(delta[:, 0], delta[:, 1], delta[:, 2], linewidth=0.2)
    plt.show()
