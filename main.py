import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from rk4 import rk4
from generate_observations import generate_data

sigma = 10
rho = 28
beta = 8 / 3


def generate_test_sequences(n, num_timesteps):
    x = np.zeros((n, num_timesteps + 1, 3))
    for i in range(n):
        x[i, 0, :] = np.random.randint(-num_timesteps*2,num_timesteps*2, size=(1,3))
        m = np.random.randint(-10, 10)
        for j in range(num_timesteps):
            x[i, j+1, :] = x[i, j, :] + m
    np.save('test_linear_gradients_integers', x)


# generate_test_sequences(5000, 500)
# observations = np.load("test_ascending_integers.npy")
# print(observations[1,:,:])


# Number of independent time series in the data set
n = 10000
# Time span of each time series
time_span = 5
# The value of c used in the imperfect model
c = 200
# The time step of the time series
time_step = 0.1
# The time step used during the rk4 integration method
integration_time_step = 0.01
# The standard deviation of the random error added to the observations
std = 0

t, observations, predictions = generate_data(n, time_span, c, time_step, integration_time_step, std)
name = f"{n}_{time_span}_{c}_{str(time_step).replace('.', '')}_{str(integration_time_step).replace('.', '')}_{std}"
np.save(f"data_no_std/test_observations_{name}", observations)
np.save(f"data_no_std/test_predictions_{name}", predictions)
np.save(f"data_no_std/test_t_{name}", t)
