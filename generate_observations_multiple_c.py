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

    t, x = t_full[0::int(step/integration_step)], x_full[0::int(step/integration_step),:]

    # standard deviation of the Gaussian error added
    num = int((final_t - initial_t) / step)
    s = x + np.random.normal(0, std, (num + 1, len(initial_x)))

    #####################
    mse = (((s - x)**2).mean(axis=0)).mean(axis=0)

    return t, s, mse


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

    if c == np.inf:
        derivative = perfect_model
    else:
        def derivative(t, x):
            # We replace x with c * np.sin(x/c) in the differential equations
            return perfect_model(t, c * np.sin(x / c))

    num = int(time_step / integration_time_step)
    t, x = rk4(initial_x, initial_t, derivative, integration_time_step, num)

    return x[-1, :]


def generate_and_predict_one(number_of_samples, time_span, c_array, time_step, integration_time_step, number_timesteps_predict, std, load_filename = None):
    
    # The scale factor of the data
    maximum_allowed = 100
    
    if load_filename is None:    
        # Generate observations and predict the last timestep for various values of c
        # Scale the data between -1 and 1
        all_s = np.zeros((number_of_samples, int(time_span / time_step) + 1, 3))
        
        # Create a dictionary that will contain all the data
        dictionary = {}

        # Create observations
        mse = np.zeros((number_of_samples))
        for i in range(number_of_samples):
            # Randomly generate initial value for x
            x0 = 10 * np.random.rand(3) + 2
            t, all_s[i,:,:], mse[i] = generate_observation(x0, 0, time_span, time_step, integration_time_step, std)
            print(i)
        
        final_mse = mse.mean(axis=0)
        print(f'final mse = {final_mse}')
    else:
        dictionary = np.load(f'{load_filename}.npy', allow_pickle=True).item()
        t = np.arange(0, time_span, time_step)
        # Scale the data back up
        all_s = dictionary["observations"]*maximum_allowed

    # Make predictions
    for c in c_array:
        print(f'c ={c}')
        predictions = np.zeros((number_of_samples, 3))
        for i in range(number_of_samples):
            predictions[i, :] = predict_imperfect(all_s[i,-(number_timesteps_predict+1), :], t[-(number_timesteps_predict+1)], c, time_step * number_timesteps_predict, integration_time_step)
        dictionary[c] = predictions

    # Scale the data
    if np.max(abs(all_s)) > maximum_allowed:
        print(f'warning, the values exceed {maximum_allowed}')
        print(f'f reach = {np.max(abs(all_s))}')
    all_s = all_s/maximum_allowed
    dictionary["observations"] = all_s

    for c in c_array:
        dictionary[c] = dictionary[c]/maximum_allowed
        
    return dictionary


# Number of independent time series in the data set
n = 10000
# Time span of each time series
time_span = 5
# The value of c used in the imperfect model
c_array = [1, 2, 5, 10]
# The time step of the time series
time_step = 0.1
# The time step used during the rk4 integration method
integration_time_step = 0.01
# The number of timesteps ahead the imperfect model predicts
number_timesteps_predict = 2
# The standard deviation of the random error added to the observations
std = 0.01 

name = f"{n}_{time_span}_{str(time_step).replace('.', '')}_{str(integration_time_step).replace('.', '')}_{number_timesteps_predict}_{str(std).replace('.', '')}"
filename = f'data_dictionaries/data_{name}'

dictionary = generate_and_predict_one(n, time_span, c_array, time_step, integration_time_step, number_timesteps_predict, std, load_filename=filename)

# Save variables
dictionary["n"] = n
dictionary["time_span"] = time_span
dictionary["time_step"] = time_step
dictionary["integration_time_step"] = integration_time_step
dictionary["number_timesteps_predict"] = number_timesteps_predict
dictionary["std"] = std

np.save(filename, dictionary)