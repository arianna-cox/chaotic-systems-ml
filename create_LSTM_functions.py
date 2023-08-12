from tensorflow import keras
from keras import layers


def create_model(type_of_model = 'MLonly', return_sequences = False):
    if type_of_model == 'MLonly':
        model = keras.Sequential()
        model.add(layers.LSTM(64, input_shape=(None, 3), return_sequences=return_sequences))
        model.add(layers.Dense(16))
        model.add(layers.Dense(3))
        return model
    if type_of_model == 'physical' or 'physical_error':
        observation_input = layers.Input(shape=(None,3), name = 'input_ob')
        observation_LSTM = layers.LSTM(64, input_shape=(None, 3), return_sequences=return_sequences)(observation_input)

        prediction_input = layers.Input(shape=(3), name = 'input_pred')
        prediction_activation = layers.Activation('linear')(prediction_input)

        merge_layers = layers.Concatenate(axis=-1)([observation_LSTM, prediction_activation])
        merge_dense = layers.Dense(16)(merge_layers)
        output = layers.Dense(3)(merge_dense)

        model = keras.Model(inputs = [observation_input, prediction_input], outputs = output)
        return model
    else:
        print('error with model name variable')


def create_callbacks(model, patience, save_filepath):

    # Intoduce early stopping to automatically train the model
    # This callback will stop the training when there is no improvement in the validation loss for 'patience' consecutive epochs
    stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

    # This call back saves the model each time the model reaches a new best validation loss
    def save_model_checkpoint(epoch, logs):
        if logs['val_loss'] < save_model_checkpoint.best_val_loss:
            save_model_checkpoint.best_val_loss = logs['val_loss']
            model.save(save_filepath)
            print('Best model saved')
    
    save_model_checkpoint.best_val_loss = float('inf')
    save_callback = keras.callbacks.LambdaCallback(on_epoch_end=save_model_checkpoint)

    return [stop_callback, save_callback]