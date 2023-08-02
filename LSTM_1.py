# import numpy as np
# import tensorflow as tf
# import keras
# from keras.models import Sequential
# from keras.layers import Dense, LSTM, Dropout
# # from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

history = compile_and_fit(lstm_model, wide_window)


#%%
