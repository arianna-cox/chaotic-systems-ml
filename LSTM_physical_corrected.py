#### QUESTION! If the observations have noise, do I use the noisy last observation as the answer or do I use the perfect last observation?

from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from load_and_prepare_data_functions import load_and_subsample_series, split_training_data
from create_LSTM_functions import create_model, create_callbacks


# Variables relating to the data you want to load
system = 'Moore'
number_of_data_points = 10000
length_of_subsequence = 20
x_transformation_type = 0
number_timesteps_predict = 1

c_array = [30,40,50,60,70,80,90,100,110,120,130,140,150,175,200,225,250,275,300,400,500,600,700,800,900,1000]

# Load the epoch dictionary
epoch_dictionary_savename = f"{system}_{number_of_data_points}_{length_of_subsequence}_{number_timesteps_predict}"
epoch_dictionary_filename = f'saved_models/{system}/timesteps_{number_timesteps_predict}/epoch_dictionary_{epoch_dictionary_savename}.npy'
epoch_dictionary = np.load(epoch_dictionary_filename , allow_pickle = True).item()

for c in c_array:
    print(f'c = {c}')

    # Load the observations
    observations, predictions = load_and_subsample_series(number_of_data_points,
                                                          system,
                                                          length_of_subsequence + number_timesteps_predict,
                                                          number_timesteps_predict = number_timesteps_predict,
                                                          x_transformation_type = x_transformation_type,
                                                          c = c)

    # Split into training and test data
    train_X, test_X, train_answer, test_answer = split_training_data(observations,
                                                                     number_timesteps_predict,
                                                                     predictions = predictions)

    print(train_X['input_pred'].shape)
    print(train_X['input_ob'].shape)
    print(test_answer.shape)
    print(train_answer.shape)
    mse = ((train_X['input_pred']-train_answer)**2).mean()
    print(f'mse = {mse}')

    # Create the architecture
    model = create_model(type_of_model = 'physical')

    model.summary()

    model.compile(loss='mse', optimizer='adam')


    # Create callbacks
    save_name = f"{system}{x_transformation_type}_c{str(c).replace('.', '')}_{number_of_data_points}_{length_of_subsequence}_{number_timesteps_predict}"
    save_filepath = f'saved_models/{system}/timesteps_{number_timesteps_predict}/x_transformation_{x_transformation_type}/physical_{save_name}.keras'
    # model = keras.models.load_model(save_filepath)
    patience = 15
    callbacks = create_callbacks(model, patience, save_filepath)

    # Training
    history = model.fit(train_X, train_answer, epochs=1000, validation_data=(test_X, test_answer), batch_size=64, callbacks=callbacks, verbose=2, shuffle=True)

    # Number of epochs needed for training
    num_epochs = len(history.history['val_loss']) - patience
    print(f'epochs = {num_epochs}')
    if f'x_transformation_{x_transformation_type}' not in epoch_dictionary[f'x_transformation_{x_transformation_type}']:
        epoch_dictionary[f'x_transformation_{x_transformation_type}'] = {}
    if 'physical' not in epoch_dictionary[f'x_transformation_{x_transformation_type}']:
        epoch_dictionary[f'x_transformation_{x_transformation_type}']['physical'] = {}
    epoch_dictionary[f'x_transformation_{x_transformation_type}']['physical'][c] = num_epochs
    
    # Save the epoch dictionary
    np.save(epoch_dictionary_filename , epoch_dictionary)
    print('saved epoch dictionary')
