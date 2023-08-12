import numpy as np

# Load the observations
def load_data(n, time_span, time_step, integration_time_step, number_timesteps_predict, std, c = None):
    load_name = f"{n}_{time_span}_{str(time_step).replace('.', '')}_{str(integration_time_step).replace('.', '')}_{number_timesteps_predict}_{str(std).replace('.', '')}"
    dictionary = np.load(f"data_dictionaries/data_{load_name}.npy", allow_pickle=True).item()
    observations = dictionary["observations"]
    if c is not None:
        predictions = dictionary[c]
        return observations, predictions
    return observations


def split_training_data(observations, number_timesteps_predict, predictions = None, predict_error = False, frac = 0.9):
    num_samples = observations.shape[0]
    cut_off = int(frac*num_samples)
    train_x = observations[:cut_off,:-number_timesteps_predict, :]
    test_x = observations[cut_off:,:-number_timesteps_predict, :]
    train_answer = observations[:cut_off, -1, :]
    test_answer = observations[cut_off:, -1, :]

    if predictions is not None:
        train_prediction = predictions[:cut_off, :]
        test_prediction = predictions[cut_off:, :]
        train_X = {"input_ob": train_x, "input_pred": train_prediction}
        test_X = {"input_ob": test_x, "input_pred": test_prediction}

        if predict_error == True:
            train_answer -= train_prediction
            test_answer -= test_prediction

        return train_X, test_X, train_answer, test_answer
    
    return train_x, test_x, train_answer, test_answer
