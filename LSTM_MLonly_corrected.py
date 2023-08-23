#### QUESTION! If the observations have noise, do I use the noisy last observation as the answer or do I use the perfect last observation?

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from load_and_prepare_data_functions import load_and_subsample_series, split_training_data
from create_LSTM_functions import create_model, create_callbacks


# Variables relating to the data you want to load
system = 'Lorentz'
number_of_data_points = 10000
length_of_subsequence = 20
list_number_timesteps_predict = [1,3,5]

for number_timesteps_predict in list_number_timesteps_predict:
    print(f'number timesteps = {number_timesteps_predict}')

    # Load the epoch dictionary
    epoch_dictionary_savename = f"{system}_{number_of_data_points}_{length_of_subsequence}_{number_timesteps_predict}"
    epoch_dictionary_filename = f'saved_models/{system}/timesteps_{number_timesteps_predict}/epoch_dictionary_corrected_{epoch_dictionary_savename}.npy'
    # epoch_dictionary = np.load(epoch_dictionary_filename , allow_pickle = True).item()

    epoch_dictionary = {}
    epoch_dictionary['MLonly'] = []
    
    # Load the observations and rearrange into smaller sequences
    observations = load_and_subsample_series(number_of_data_points,
                                            system,
                                            length_of_subsequence + number_timesteps_predict)
    print(observations.shape)

    # Split into training and test data
    train_x, test_x, train_answer, test_answer = split_training_data(observations, number_timesteps_predict)
    print(train_x.shape)
    print(test_x.shape)
    print(test_answer.shape)
    print(train_answer.shape)
    
    #######
    # name = f"{system}_{number_of_data_points}"
    # filename = f'data_dictionaries/data_std0001_{name}.npy'
    # ######
    for i in range(5):
        # Create the architecture
        model = create_model(type_of_model = 'MLonly')

        model.summary()

        # optimizer = keras.optimizers.Adam(0.001)
        # optimizer.learning_rate.assign(0.01)

        model.compile(loss='mse', optimizer='adam')

        # Create callbacks
        save_name = f"{system}_{number_of_data_points}_{length_of_subsequence}_{number_timesteps_predict}"
        save_filepath = f'saved_models/{system}/timesteps_{number_timesteps_predict}/MLonly_{save_name}.keras'
        patience = 15
        callbacks = create_callbacks(model, patience, save_filepath)

        # Training
        history = model.fit(train_x, train_answer, epochs=1000, validation_data=(test_x, test_answer), batch_size=64, callbacks=callbacks, verbose=2, shuffle=True)

        # Number of epochs needed for training
        num_epochs = len(history.history['val_loss']) - patience
        print(f'epochs = {num_epochs}')

        epoch_dictionary['MLonly'].append(num_epochs)
    
    np.save(epoch_dictionary_filename, epoch_dictionary)