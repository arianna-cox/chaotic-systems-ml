#### QUESTION! If the observations have noise, do I use the noisy last observation as the answer or do I use the perfect last observation?

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from load_and_prepare_data_functions import load_data, split_training_data
from create_LSTM_functions import create_model, create_callbacks


# Variables relating to the data you want to load
system = 'Rossler'
x_transformation_type = 0
n = 10000
time_span = 5
time_step = 0.1
integration_time_step = 0.01
number_timesteps_predict = 5
std = 0.01

# Load the observations
observations = load_data(n, time_span, time_step, integration_time_step, number_timesteps_predict, std, system = system, x_transformation_type = x_transformation_type)

# Split into training and test data

train_x, test_x, train_answer, test_answer = split_training_data(observations, number_timesteps_predict)

# Create the architecture
model = create_model(type_of_model = 'MLonly')

model.summary()

# optimizer = keras.optimizers.Adam(0.001)
# optimizer.learning_rate.assign(0.01)

model.compile(loss='mse', optimizer='adam')

# Create callbacks
save_name = f"{system}{x_transformation_type}_{n}_{time_span}_{str(time_step).replace('.', '')}_{str(integration_time_step).replace('.', '')}_{number_timesteps_predict}_{str(std).replace('.', '')}"
save_filepath = f'saved_models/{system}{x_transformation_type}/timesteps_{number_timesteps_predict}/MLonly_{save_name}.keras'
patience = 15
callbacks = create_callbacks(model, patience, save_filepath)

# Training
history = model.fit(train_x, train_answer, epochs=1000, validation_data=(test_x, test_answer), batch_size=64, callbacks=callbacks, verbose=2, shuffle=True)

# Number of epochs needed for training
num_epochs = len(history.history['val_loss']) - patience
print(f'epochs = {num_epochs}')

# Load the epoch dictionary
epoch_dictionary_savename = f"{system}{x_transformation_type}_{n}_{time_span}_{str(time_step).replace('.', '')}_{str(integration_time_step).replace('.', '')}_{number_timesteps_predict}_{str(std).replace('.', '')}"
epoch_dictionary_filename = f'saved_models/{system}{x_transformation_type}/timesteps_{number_timesteps_predict}/epoch_dictionary_{epoch_dictionary_savename}.npy'
epoch_dictionary = np.load(epoch_dictionary_filename , allow_pickle = True).item()

# epoch_dictionary = {}
# epoch_dictionary['physical'] = {}
# epoch_dictionary['physical_error'] = {}
epoch_dictionary['MLonly'] = num_epochs
np.save(epoch_dictionary_filename, epoch_dictionary)