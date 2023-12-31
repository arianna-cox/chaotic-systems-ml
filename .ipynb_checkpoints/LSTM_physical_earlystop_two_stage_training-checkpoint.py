#### QUESTION! If the observations have noise, do I use the noisy last observation as the answer or do I use the perfect last observation?

from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from load_and_prepare_data_functions import load_data, split_training_data
from create_LSTM_functions import create_model, create_callbacks


# Variables relating to the data you want to load
n = 10000
time_span = 5
time_step = 0.1
integration_time_step = 0.01
number_timesteps_predict = 2
std = 0.01
c_array = [500]

for c in c_array:

    # Load the observations
    observations, predictions = load_data(n, time_span, time_step, integration_time_step, number_timesteps_predict, std, c = c)

    # Split into training and test data
    train_X_random, test_X_random, train_X, test_X, train_answer, test_answer = split_training_data(observations, number_timesteps_predict, predictions = predictions, random_predictions = True)

    # Create the architecture
    model = create_model(type_of_model = 'physical')

    model.summary()

    optimizer = keras.optimizers.Adam(0.001)
    optimizer.learning_rate.assign(0.01)

    model.compile(loss='mse', optimizer=optimizer)


    # Create callbacks
    save_name = f"c{str(c).replace('.', '')}_{n}_{time_span}_{str(time_step).replace('.', '')}_{str(integration_time_step).replace('.', '')}_{number_timesteps_predict}_{str(std).replace('.', '')}"
    save_filepath = f'saved_models/timesteps_{number_timesteps_predict}/physical_two_stage_training_{save_name}.keras'
    # model = keras.models.load_model(save_filepath)
    patience = 15
    callbacks = create_callbacks(model, patience, save_filepath)

    # Training

    history = model.fit(train_X_random, train_answer, epochs=1000, validation_data=(test_X_random, test_answer), batch_size=64, callbacks=callbacks, verbose=2, shuffle=True)

    # Change the learning rate
    optimizer.learning_rate.assign(0.001)
    model.compile(loss='mse', optimizer=optimizer)
    
    history = model.fit(train_X, train_answer, epochs=1000, validation_data=(test_X, test_answer), batch_size=64, callbacks=callbacks, verbose=2, shuffle=True)

    # Print how many epochs
    num_epochs = len(history.history['val_loss'])
    print(f'epochs = {num_epochs}')