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


def generate_observation_and_discard(initial_x, initial_t, final_t, step, integration_step, std, discard = 10000, perfect_model = perfect_model_Lorentz):
    """
    Solves the Lorentz63 ODE and adds noise (Gaussian error) to the solution
    :param initial_x: initial position
    :param initial_t: initial time
    :param final_t: final time
    :param step: Fixed time step for integration
    :param std: standard deviation of the Gaussian error
    :return: numpy arrays of time and observations
    """

    # Initialize by running many steps that are discarded so that x0 is on the attractor
    num_discard_integration = int(discard * (step/integration_step))
    print(f'discard = {num_discard_integration}')
    x0 = rk4(initial_x, 0, perfect_model, integration_step, num_discard_integration)[1][-1, :]
    print('Initial x0 reached')
    print(f'x0 = {x0}')
    
    num_integration = int((final_t - initial_t)/integration_step)
    t_full, x_full = rk4(x0, initial_t, perfect_model, integration_step, num_integration)

    t = t_full[0::int(step/integration_step)]
    x = x_full[0::int(step/integration_step),:]

    # Gaussian error added
    num = int((final_t - initial_t) / step)
    s = x + np.random.normal(0, std, (num + 1, len(initial_x)))

    return t, x, s


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


def generate_series_and_predict(time_span, time_step, integration_time_step, std, list_timesteps_predict = [], c_array = [], load_filename = None, perfect_model = perfect_model_Lorentz, x_transformation_types = [0]):
    # Generate observations and predict the last timestep for various values of c
    # Scale the data between -1 and 1
    # The scale factor of the data
    maximum_allowed = 100
    
    if load_filename is None:    
        # Create a dictionary that will contain all the data
        dictionary = {}

        # Randomly generate initial value for x near zero
        x0 = 0.1 * np.random.rand(3)
        
        # Generate the time series
        t, x, s = generate_observation_and_discard(x0, 0, time_span, time_step, integration_time_step, std, perfect_model = perfect_model)
    
    else:
        dictionary = np.load(f'{load_filename}.npy', allow_pickle=True).item()
        t = np.arange(0, time_span + time_step, time_step)
        # Scale the data back up
        s = dictionary["observations"]*maximum_allowed

    # Make predictions
    for timesteps_predict in list_timesteps_predict:
        print(f'timesteps predicted forward = {timesteps_predict}')
        if f'timesteps_{timesteps_predict}' not in dictionary:
            dictionary[f'timesteps_{timesteps_predict}'] = {}
        for type in x_transformation_types:
            print(f'x_transformation_type = {type}')
            if f'x_transformation_{type}' not in dictionary[f'timesteps_{timesteps_predict}']:
                dictionary[f'timesteps_{timesteps_predict}'][f'x_transformation_{type}'] = {}
            for c in c_array[type]:
                print(f'c ={c}')
                predictions = np.zeros(s.shape)
                for i in range(len(predictions[:,0])):
                    # The ith entry of prediction corresponds to a prediction timesteps_predict into the future using the ith entry of s
                    predictions[i, :] = predict_imperfect(s[i, :],
                                                          t[i],
                                                          c,
                                                          time_step * timesteps_predict,
                                                          integration_time_step,
                                                          perfect_model = perfect_model,
                                                          x_transformation_type = type)
                dictionary[f'timesteps_{timesteps_predict}'][f'x_transformation_{type}'][c] = predictions

    # Scale the data
    if np.max(abs(s)) > maximum_allowed:
        print(f'warning, the values exceed {maximum_allowed}')
        print(f'maximum reached = {np.max(abs(s))}')
    if load_filename is None:  
        dictionary["observations"] = s/maximum_allowed
        dictionary["x"] = x/maximum_allowed

    for timesteps_predict in list_timesteps_predict:
        for x_transformation_type in x_transformation_types:
            for c in c_array[x_transformation_type]:
                dictionary[f'timesteps_{timesteps_predict}'][f'x_transformation_{x_transformation_type}'][c] /= maximum_allowed
        
    return dictionary

# Time span of each time series
time_span = 1000
# The value of c used in the imperfect model
c_array_0 = [30,40,50,60,70,80,90,100,110,120,130,140,150,175,200,225,250,275,300,400,500,600,700,800,900,1000]
c_array = [c_array_0]
# The time step of the time series
time_step = 0.1
# The time step used during the rk4 integration method
integration_time_step = 0.01
# The number of timesteps ahead the imperfect model predicts
number_timesteps_predict = [1,2,3,5,10]
# The standard deviation of the random error added to the observations
std = 0.01

system = 'Lorentz'
perfect_model_system = perfect_model_Lorentz
x_transformation_types = [0]

number_of_data_points = int(time_span/time_step)
print(f'number of data points kept = {number_of_data_points}')
name = f"{system}_{number_of_data_points}"
filename = f'data_dictionaries/data_testing_{name}'

dictionary = generate_series_and_predict(time_span,
                                      time_step,
                                      integration_time_step,
                                      std,
                                      list_timesteps_predict = number_timesteps_predict,
                                      c_array = c_array,
                                      load_filename = None,
                                      perfect_model = perfect_model_system, 
                                      x_transformation_types = x_transformation_types)

# Save variables
dictionary["time_span"] = time_span
dictionary["time_step"] = time_step
dictionary["integration_time_step"] = integration_time_step
dictionary["std"] = std

np.save(filename, dictionary)
