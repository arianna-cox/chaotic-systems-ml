from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from load_and_prepare_data_functions import load_and_subsample_series, split_training_data
from create_LSTM_functions import create_model, create_callbacks


# Variables relating to the data you want to load
system = 'Lorentz'
number_of_data_points = 10000
length_of_subsequence = 20
x_transformation_type = 0
number_timesteps_predict = 10
repetitions = 5

c = 50

#######
name = f"{system}_{number_of_data_points}"
filename = f'data_dictionaries/data_testing_{name}.npy'
#####


# Load the observations
observations, predictions = load_and_subsample_series(number_of_data_points,
                                                        system,
                                                        length_of_subsequence + number_timesteps_predict,
                                                        number_timesteps_predict = number_timesteps_predict,
                                                        x_transformation_type = x_transformation_type,
                                                        c = c, filename = filename)

# Split into training and test data
train_X_random, test_X_random, train_X, test_X, train_answer, test_answer = split_training_data(observations,
                                                                    number_timesteps_predict,
                                                                    predictions = predictions,
                                                                    predict_error = True,
                                                                    random_predictions = True)

# Train the random ML physical error models
for i in range(repetitions):

    # Create the architecture
    model = create_model(type_of_model = 'physical_error')

    model.summary()

    model.compile(loss='mse', optimizer='adam')


    # Create callbacks
    save_name = f"{system}{x_transformation_type}_c{str(c).replace('.','')}_{number_of_data_points}_{length_of_subsequence}_{number_timesteps_predict}"
    save_filepath = f'saved_models/random/physical_error_random_number{i}_{save_name}'
    # model = keras.models.load_model(save_filepath)
    patience = 15
    callbacks = create_callbacks(model, patience, save_filepath)

    # Training
    history = model.fit(train_X_random, train_answer, epochs=1000, validation_data=(test_X, test_answer), batch_size=64, callbacks=callbacks, verbose=2, shuffle=True)

    # Number of epochs needed for training
    num_epochs = len(history.history['val_loss']) - patience
    print(f'epochs = {num_epochs}')

for i in range(repetitions):

    # Create the architecture
    model = create_model(type_of_model = 'physical_error')

    model.summary()

    model.compile(loss='mse', optimizer='adam')


    # Create callbacks
    save_name = f"{system}{x_transformation_type}_c{str(c).replace('.','')}_{number_of_data_points}_{length_of_subsequence}_{number_timesteps_predict}"
    save_filepath = f'saved_models/random/physical_error_notrandom_number{i}_{save_name}'
    # model = keras.models.load_model(save_filepath)
    patience = 15
    callbacks = create_callbacks(model, patience, save_filepath)

    # Training
    history = model.fit(train_X, train_answer, epochs=1000, validation_data=(test_X, test_answer), batch_size=64, callbacks=callbacks, verbose=2, shuffle=True)

    # Number of epochs needed for training
    num_epochs = len(history.history['val_loss']) - patience
    print(f'epochs = {num_epochs}')