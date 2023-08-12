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

for c in [10,50,100,200,500]:

    # Load the observations
    observations, predictions = load_data(n, time_span, time_step, integration_time_step, number_timesteps_predict, std, c = c)

    # Split into training and test data
    train_X, test_X, train_answer, test_answer = split_training_data(observations, number_timesteps_predict, predictions = predictions, predict_error = True)

    # Create the architecture
    model = create_model(type_of_model = 'physical_error')

    model.summary()

    model.compile(loss='mse', optimizer='adam')


    # Create callbacks
    save_name = f"c{c}_{n}_{time_span}_{str(time_step).replace('.', '')}_{str(integration_time_step).replace('.', '')}_{number_timesteps_predict}_{str(std).replace('.', '')}"
    save_filepath = f'saved_models/timesteps_{number_timesteps_predict}/physical_model_{save_name}.keras'
    # model = keras.models.load_model(save_filepath)
    patience = 15
    callbacks = create_callbacks(model, patience, save_filepath)

    # Training
    history = model.fit(train_X, train_answer, epochs=300, validation_data=(test_X, test_answer), batch_size=32, callbacks=callbacks, verbose=2, shuffle=True)

    # Print how many epochs
    num_epochs = len(history.history['val_loss'])
    print(f'epochs = {num_epochs}')

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
