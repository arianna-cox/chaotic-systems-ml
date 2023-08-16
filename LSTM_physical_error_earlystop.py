#### QUESTION! If the observations have noise, do I use the noisy last observation as the answer or do I use the perfect last observation?

from tensorflow import keras
from keras import layers
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
c_array = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140 ,150, 175, 200, 225, 250, 300, 400, 500, 600, 700, 800, 900, 1000]

# Load the epoch dictionary
epoch_dictionary_savename = f"{system}{x_transformation_type}_{n}_{time_span}_{str(time_step).replace('.', '')}_{str(integration_time_step).replace('.', '')}_{number_timesteps_predict}_{str(std).replace('.', '')}"
epoch_dictionary_filename = f'saved_models/{system}{x_transformation_type}/timesteps_{number_timesteps_predict}/epoch_dictionary_{epoch_dictionary_savename}.npy'
epoch_dictionary = np.load(epoch_dictionary_filename , allow_pickle = True).item()

for c in c_array:

    # Load the observations
    observations, predictions = load_data(n, time_span, time_step, integration_time_step, number_timesteps_predict, std, c = c, system = system, x_transformation_type = x_transformation_type)

    # Split into training and test data
    train_X, test_X, train_answer, test_answer = split_training_data(observations, number_timesteps_predict, predictions = predictions, predict_error = True)

    # Create the architecture
    model = create_model(type_of_model = 'physical_error')

    model.summary()

    model.compile(loss='mse', optimizer='adam')


    # Create callbacks
    save_name = f"{system}{x_transformation_type}_c{str(c).replace('.', '')}_{n}_{time_span}_{str(time_step).replace('.', '')}_{str(integration_time_step).replace('.', '')}_{number_timesteps_predict}_{str(std).replace('.', '')}"
    save_filepath = f'saved_models/{system}{x_transformation_type}/timesteps_{number_timesteps_predict}/physical_error_{save_name}.keras'
    patience = 15
    callbacks = create_callbacks(model, patience, save_filepath)

    # Training
    history = model.fit(train_X, train_answer, epochs=1000, validation_data=(test_X, test_answer), batch_size=64, callbacks=callbacks, verbose=2, shuffle=True)

    # Number of epochs needed for training
    num_epochs = len(history.history['val_loss']) - patience
    print(f'epochs = {num_epochs}')
    epoch_dictionary['physical_error'][c] = num_epochs
    
    # Save the epoch dictionary
    np.save(epoch_dictionary_filename , epoch_dictionary)
    print('saved epoch dictionary')

# # Plot history
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='val')
# plt.yscale('log')
# plt.legend()
# plt.show()

# Plot some predictions of the model vs the answer vs the imperfect model
# a = 0
# b = 2
# for i in range(1,10,1):
#     plt.plot(test_x[i,-2:,0], test_x[i,-2:,1], label = 'observation_data', color = 'g')
#     plt.scatter(test_answer[i,a], test_answer[i,b], label = 'answer', s=5, color = 'r')
#     plt.scatter(test_prediction[i,a], test_prediction[i,b], label = 'prediction', s=10, color = 'y', marker ='o')
#     plt.legend()
#     plt.show()
