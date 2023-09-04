import numpy as np

# Load the observations
def load_data(n, time_span, time_step, integration_time_step, number_timesteps_predict, std, c = None, system = 'Lorentz', x_transformation_type = 0):
    load_name = f"{system}_{n}_{time_span}_{str(time_step).replace('.', '')}_{str(integration_time_step).replace('.', '')}_{number_timesteps_predict}_{str(std).replace('.', '')}"
    dictionary = np.load(f"data_dictionaries/data_{load_name}.npy", allow_pickle=True).item()
    observations = dictionary["observations"]
    if c is not None:
        predictions = dictionary[f'x_transformation_{x_transformation_type}'][c]
        return observations, predictions
    return observations

def load_and_subsample_series(number_of_data_points, system, length_of_subsequence, number_timesteps_predict = 1, x_transformation_type = 0, c = None, filename = None):
    if filename == None:
        load_name = f"data_dictionaries/data_{system}_{number_of_data_points}.npy"
    else:
        load_name = filename
    dictionary = np.load(load_name, allow_pickle=True).item()
    s = dictionary["observations"]

    length_of_timeseries, x_shape = s.shape
    number_of_samples = length_of_timeseries - length_of_subsequence + 1
    observations_subsampled = np.zeros((number_of_samples, length_of_subsequence, x_shape))
    for i in range(number_of_samples):
        observations_subsampled[i,:,:] = s[i:i+length_of_subsequence, :]
            
    if c is not None:
        predictions = dictionary[f'timesteps_{number_timesteps_predict}'][f'x_transformation_{x_transformation_type}'][c]
        predictions_subsampled = np.zeros((number_of_samples, x_shape))
        for i in range(number_of_samples):
            # The prediction is made from entry number (i+length_of_subsequence-1)-number_timesteps_predict) to predict observation (i+length_of_subsequence-1)
            predictions_subsampled[i,:] = predictions[(i+length_of_subsequence-1)-number_timesteps_predict, :]
        return observations_subsampled, predictions_subsampled
    return observations_subsampled


def split_training_data(observations, number_timesteps_predict, predictions = None, predict_error = False, random_predictions = False, return_all_lead_times = False, frac = 0.9):
    num_samples = observations.shape[0]
    cut_off = int(frac*num_samples)
    train_x = observations[:cut_off,:-number_timesteps_predict, :]
    test_x = observations[cut_off:,:-number_timesteps_predict, :]
    train_answer = observations[:cut_off, -1, :]
    test_answer = observations[cut_off:, -1, :]

    if return_all_lead_times == True:
        train_answer = observations[:cut_off, -number_timesteps_predict:, :]
        test_answer = observations[cut_off:, -number_timesteps_predict:, :]

    if predictions is not None:
        train_prediction = predictions[:cut_off, :]
        test_prediction = predictions[cut_off:, :]
        train_X = {"input_ob": train_x, "input_pred": train_prediction}
        test_X = {"input_ob": test_x, "input_pred": test_prediction}

        if predict_error == True:
            train_answer -= train_prediction
            test_answer -= test_prediction

        if random_predictions == True:
            train_pred_random = np.random.rand(*train_prediction.shape)
            test_pred_random = np.random.rand(*test_prediction.shape)
            train_X_random = {"input_ob": train_x, "input_pred": train_pred_random}
            test_X_random = {"input_ob": test_x, "input_pred": test_pred_random}
            return train_X_random, test_X_random, train_X, test_X, train_answer, test_answer

        return train_X, test_X, train_answer, test_answer
    
    return train_x, test_x, train_answer, test_answer
