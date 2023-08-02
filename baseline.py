from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

return_sequences = False
n = 10000
time_span = 5
c = 100
time_step = 0.1
integration_time_step = 0.01
std = 0

# Load the observations
name = f"{n}_{time_span}_{c}_{str(time_step).replace('.', '')}_{str(integration_time_step).replace('.', '')}_{std}"
observations = np.load(f"data_no_std/test_observations_{name}.npy")
# predictions = np.load(f"data_no_std/test_predictions_{name}.npy")
print(f"observation array shape = {observations.shape}")

# Scale the data
print(np.max(np.abs(observations)))
observations_scaled = observations/np.max(np.abs(observations))

# Split into training and test data

frac = 0.9
num_samples = observations_scaled.shape[0]
train_x = observations_scaled[:int(frac*num_samples),:-1, :]
test_x = observations_scaled[int(frac*num_samples):,:-1, :]
if return_sequences:
    train_answer = observations_scaled[:int(frac*num_samples), 1:, :]
    test_answer = observations_scaled[int(frac*num_samples):, 1:, :]
else:
    train_answer = observations_scaled[:int(frac*num_samples), -1, :]
    test_answer = observations_scaled[int(frac*num_samples):, -1, :]
print(f"shape of training data = {train_x.shape}")
print(f"shape of training answers = {train_answer.shape}")



# Create the architecture
model = keras.Sequential()
model.add(layers.LSTM(64, input_shape=(None, 3), return_sequences=return_sequences))
# model.add(layers.Flatten(input_shape=(401, 3)))
# model.add(layers.Dense(64))
model.add(layers.Dense(16))
model.add(layers.Dense(3))
model.summary()

model.compile(loss='mse', optimizer='adam')

# Train the model
history = model.fit(train_x, train_answer, epochs=15, batch_size=32, validation_data=(test_x, test_answer), verbose=2, shuffle=True)

# Plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.show()

ML_predictions = model.predict(test_x)
print(ML_predictions.shape)

for i in range(10):
    plt.scatter(ML_predictions[i,0], ML_predictions[i,1], label = 'ML_prediction', s=10, marker ='x')
    plt.plot(test_x[i,:,0], test_x[i,:,1], label = 'data')
    plt.scatter(test_answer[i,0], test_answer[i,1], label = 'answer', s=10)
    plt.legend()
    plt.show()

