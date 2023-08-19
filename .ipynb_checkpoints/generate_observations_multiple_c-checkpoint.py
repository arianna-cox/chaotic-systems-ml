import numpy as np
from matplotlib import pyplot as plt
from rk4 import rk4

def perfect_model_Lorentz(t, x):
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


def perfect_model_Rossler(t, x):
    """
    Returns dx/dt for the Rossler attractor
    :param t: time
    :param x: position
    :return: dx/dt is expressed as a numpy vector
    """
    # Define the parameters of the Rossler attractor
    a = 0.2
    b = 0.2
    c = 5.7
    return np.array([-x[1] - x[2], x[0] + a * x[1], b + x[2] * (x[0] - c)])


def generate_observation(initial_x, initial_t, final_t, step, integration_step, std, perfect_model = perfect_model_Lorentz):
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

    # Gaussian error added
    num = int((final_t - initial_t) / step)
    s = x + np.random.normal(0, std, (num + 1, len(initial_x)))

    mse = (((s - x)**2).mean(axis=0)).mean(axis=0)

    return t, s, mse


def predict_imperfect(initial_x, initial_t, c, time_step, integration_time_step,
                      perfect_model = perfect_model_Lorentz,
                      x_transformation_type = 0):
    """
    Solves the ODE dx/dt = f(t,x) with f given by the function derivative below
    :param initial_x: Initial position
    :param initial_t: Initial time
    :param c: Controls the inaccuracy of the model
    :param time_step: Fixed time step
    :param integration_time_step: Time step used for the Runge-Kutta integration
    :param perfect_model: The chaotic system of differential equations to solve
    :param x_transformation_type: The type corresponds to a function that transforms the x coordinate as a continuous function of c to make the perfect model into an imperfect model.
    :return: A numpy vector of position at time = initial_t + time_step
    """

    # Find the corresponding x transformation
    if x_transformation_type == 0:
        x_transformation = lambda X, C : C * np.sin(X/C)
    elif x_transformation_type == 1:
        x_transformation = lambda X, C : 0
    elif x_transformation_type == 2:
        x_transformation = lambda X, C : 0
    
    if c == np.inf:
        derivative = perfect_model
    else:
        def derivative(t, x):
            # We replace x with x_transformation(x,c) in the differential equations
            return perfect_model(t, x_transformation(x,c))

    num = int(time_step / integration_time_step)
    t, x = rk4(initial_x, initial_t, derivative, integration_time_step, num)

    return x[-1, :]


def generate_and_predict_one(number_of_samples, time_span, c_array, time_step, integration_time_step, number_timesteps_predict, std, load_filename = None, perfect_model = perfect_model_Lorentz, x_transformation_types = [0]):
    
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
            # x0 = 10 * np.random.rand(3) + 2
            # Randomly generate initial value for x uniformly in the space [-10, 10] for all of x, y and z
            x0 = 20 * (np.random.rand(3) - 0.5)
            if perfect_model == perfect_model_Rossler:
                x0[2] = np.random.rand(1)
            t, all_s[i,:,:], mse[i] = generate_observation(x0, 0, time_span, time_step, integration_time_step, std, perfect_model = perfect_model)
            print(i)
        
        final_mse = mse.mean(axis=0)
        print(f'final mse = {final_mse}')
    else:
        dictionary = np.load(f'{load_filename}.npy', allow_pickle=True).item()
        t = np.arange(0, time_span, time_step)
        # Scale the data back up
        all_s = dictionary["observations"]*maximum_allowed

    # Make predictions
    for type in x_transformation_types:
        print(f'x_transformation_type = {type}')
        if f'x_transformation_{type}' not in dictionary:
            dictionary[f'x_transformation_{type}'] = {}
        for c in c_array[type]:
            print(f'c ={c}')
            predictions = np.zeros((number_of_samples, 3))
            for i in range(number_of_samples):
                predictions[i, :] = predict_imperfect(all_s[i,-(number_timesteps_predict+1), :],
                                                      t[-(number_timesteps_predict+1)],
                                                      c,
                                                      time_step * number_timesteps_predict,
                                                      integration_time_step,
                                                      perfect_model = perfect_model,
                                                      x_transformation_type = type)
            dictionary[f'x_transformation_{type}'][c] = predictions

    # Scale the data
    if np.max(abs(all_s)) > maximum_allowed:
        print(f'warning, the values exceed {maximum_allowed}')
        print(f'maximum reached = {np.max(abs(all_s))}')
    all_s = all_s/maximum_allowed
    dictionary["observations"] = all_s
    
    for x_transformation_type in x_transformation_types:
        for c in c_array[x_transformation_type]:
            dictionary[f'x_transformation_{x_transformation_type}'][c] = dictionary[f'x_transformation_{x_transformation_type}'][c]/maximum_allowed
        
    return dictionary


# Number of independent time series in the data set
n = 10000
# Time span of each time series
time_span = 5
# The value of c used in the imperfect model
c_array_0 = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140 ,150, 175, 200, 225, 250, 300, 400, 500, 600, 700, 800, 900, 1000, np.inf]
c_array = [c_array_0]
# The time step of the time series
time_step = 0.1
# The time step used during the rk4 integration method
integration_time_step = 0.01
# The number of timesteps ahead the imperfect model predicts
number_timesteps_predict = 10
# The standard deviation of the random error added to the observations
std = 0.01

system = 'Rossler'
perfect_model_system = perfect_model_Rossler
x_transformation_types = [0]

name = f"{system}_{n}_{time_span}_{str(time_step).replace('.', '')}_{str(integration_time_step).replace('.', '')}_{number_timesteps_predict}_{str(std).replace('.', '')}"
filename = f'data_dictionaries/data_{name}'

dictionary = generate_and_predict_one(n, time_span, c_array, time_step, integration_time_step, number_timesteps_predict, std, perfect_model = perfect_model_system, x_transformation_types = x_transformation_types)

# Save variables
dictionary["n"] = n
dictionary["time_span"] = time_span
dictionary["time_step"] = time_step
dictionary["integration_time_step"] = integration_time_step
dictionary["number_timesteps_predict"] = number_timesteps_predict
dictionary["std"] = std

np.save(filename, dictionary)
