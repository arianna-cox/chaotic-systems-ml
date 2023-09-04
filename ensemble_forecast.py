from tensorflow import keras
from keras import layers
import numpy as np
from load_and_prepare_data_functions import load_and_subsample_series, split_training_data
from generate_observations_corrected import perfect_model_Lorentz, predict_imperfect
import scipy.integrate

# Variables relating to the data you want to load
system = 'Lorentz'
# system_model = perfect_model_Lorentz
x_transformation_type = 0
number_of_data_points = 10000
length_of_subsequence = 20
number_timesteps_predict = 10

c_array = [30,40,50,60,70,80,90,100,110,120,130,140,150,175,200,225,250,275,300,400,500,600,700,800,900,1000]
c_array = [50,100,150,200,500,1000]

name = f"{system}_{number_of_data_points}"
test_filename = f'data_dictionaries/data_testing_{name}.npy'

# Load the observations and rearrange into smaller sequences
observations = load_and_subsample_series(number_of_data_points,
                                        system,
                                        length_of_subsequence + number_timesteps_predict,
                                        filename = test_filename)
print(observations.shape)

# Split into training and test data
train_x, test_x, train_answers, test_answers = split_training_data(observations, number_timesteps_predict, return_all_lead_times = True, frac = 0)
print(train_x.shape)
print(test_x.shape)
print(test_answers.shape)
print(train_answers.shape)

def create_ensemble(sequence_list, number_of_perturbed_sequences, c_array, number_timesteps_predict_list, time_step = 0.1, integration_timestep = 0.01, std = 0.01, maximum_allowed = 100, system_model = perfect_model_Lorentz):
    # Scale up and perturb the sequence by a Gaussian with standard deviation of std.
    # Then predict using the imperfect model for values of c in c_array, for each lead time.
    # Return the ensembles and the imperfect predictions of the ensemble in a dictionary.
    ensemble_dictionary = {}
    perturbed_sequences = np.zeros((len(sequence_list), number_of_perturbed_sequences, length_of_subsequence, 3))
    
    for num, sequence in enumerate(sequence_list):
        for i in range(number_of_perturbed_sequences):
            perturbed_sequences[num,i,:,:] = sequence * maximum_allowed + (np.random.normal(0, std, (length_of_subsequence, 3)))
    
    for c in c_array:
        ensemble_dictionary[c] = {}
        new_imperfect_predictions = np.zeros((len(sequence_list), number_of_perturbed_sequences, len(number_timesteps_predict_list), 3))
        for num, sequence in enumerate(sequence_list):
            for i in range(number_of_perturbed_sequences):
                for j, lead_time in enumerate(number_timesteps_predict_list):
                    new_imperfect_predictions[num,i,j,:] = predict_imperfect(perturbed_sequences[num, i,-1,:], 0, c, time_step * lead_time, integration_timestep, perfect_model = system_model)
        
        ensemble_dictionary[c]['imperfect_predictions'] = new_imperfect_predictions/maximum_allowed

    ensemble_dictionary['original_sequences'] = sequence_list
    ensemble_dictionary['lead_times'] = number_timesteps_predict_list
    ensemble_dictionary['perturbed_initial_conditions'] = perturbed_sequences/maximum_allowed

    return ensemble_dictionary

def sample_mean_and_covariance(ensemble_forecast):
    # Determine the sample mean and covariance of the ensemble forecast assuming a multivariate normal distribution
    n, dim = ensemble_forecast.shape
    ensemble_forecast = np.reshape(ensemble_forecast, (n, dim, 1))
    mu = np.sum(ensemble_forecast, axis=0)/n
    sigma = np.zeros((dim, dim))
    for i in range(n):
        sigma += np.matmul((ensemble_forecast[i,:] - mu), np.transpose(ensemble_forecast[i,:] - mu))/(n-1)
    return mu, sigma

def ignorance_score(ensemble_forecast, true_observation):
    mu, sigma = sample_mean_and_covariance(ensemble_forecast)
    def minus_log_p(x):
        # Probability density function of a Gaussian
        # There are k dimensions 
        k = len(x)
        x = np.reshape(x, (k,1))
        return -((np.matmul(np.matmul(np.transpose(x - mu), np.linalg.inv(sigma)), (x-mu)))/2)/np.sqrt(((2*np.pi)**k)*np.linalg.det(sigma))
    return minus_log_p(true_observation)

def naive_linear_score(ensemble_forecasts, true_observations):
    score = 0
    for num in range(ensemble_forecasts.shape[0]):
        mu, sigma = sample_mean_and_covariance(ensemble_forecasts[num,:,:])
        def p(x):
            # Probability density function of a Gaussian
            # There are k dimensions 
            k = len(x)
            x = np.reshape(x, (k,1))
            return np.exp(-(np.matmul(np.matmul(np.transpose(x - mu), np.linalg.inv(sigma)), (x-mu)))/2)/np.sqrt(((2*np.pi)**k)*np.linalg.det(sigma))
        score += - p(true_observations[num,:])
    return score / ensemble_forecasts.shape[0]

def proper_linear_score(ensemble_forecasts, true_observations):
    score = 0
    for num in range(ensemble_forecasts.shape[0]):
        mu, sigma = sample_mean_and_covariance(ensemble_forecasts[num,:,:])
        # The dimension of the ensemble members is
        k = ensemble_forecasts.shape[-1]
        def p(x):
            # Probability density function of a Gaussian
            x = np.reshape(x, (k,1))
            return np.exp(-(np.matmul(np.matmul(np.transpose(x - mu), np.linalg.inv(sigma)), (x-mu)))/2)/np.sqrt(((2*np.pi)**k)*np.linalg.det(sigma))
        # def integrand(x, y, z):
            # return p(np.array([x,y,z]))**2
        # integral = scipy.integrate.tplquad(integrand, -np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf)
        integral = 1/np.sqrt(((4*np.pi)**k)*np.linalg.det(sigma))
        score += integral - 2 * p(true_observations[num,:])
    empirical_skill = score / ensemble_forecasts.shape[0]
    return empirical_skill

# Define the sequences to be used for the ensemble forecast
test_sequences = test_x[::100,:,:]
print(f'hi = {test_sequences.shape}')
test_sequence_answers = test_answers[::100,:,:]
number_of_perturbed_sequences = 50
number_timesteps_predict_list = [1,2,3,5,10]
std = 0.01

# # Generate the forecast ensemble
# ensemble_dictionary = create_ensemble(test_sequences, number_of_perturbed_sequences, c_array, number_timesteps_predict_list, std = std)

# # Pick out the oberservational answers corresponding to the lead times
# observation_answers = np.zeros((len(test_sequences), len(number_timesteps_predict_list), 3))
# for j, lead_time in enumerate(number_timesteps_predict_list):
#     observation_answers[:,j,:] = test_sequence_answers[:, lead_time-1, :]
# ensemble_dictionary['observation_answers'] = observation_answers

# # Find predictions for the  ML only model
# ensemble_forecast_ML_only = np.zeros((len(test_sequences), number_of_perturbed_sequences, len(number_timesteps_predict_list), 3))
# for j, lead_time in enumerate(number_timesteps_predict_list):
#     # Load the ML only model
#     trunctated_filename = f'saved_models/{system}/timesteps_{lead_time}'
#     ML_only_model = keras.models.load_model(f'{trunctated_filename}/MLonly_{system}_{number_of_data_points}_{length_of_subsequence}_{lead_time}.keras')
    
#     # Make the ML model predictions
#     for num in range(len(test_sequences)):
#         ensemble_forecast_ML_only[num,:,j,:] = ML_only_model.predict(ensemble_dictionary['perturbed_initial_conditions'][num,:,:,:])
# ensemble_dictionary['ML_only_forecast'] = ensemble_forecast_ML_only


# # Find predictions for the hybrid ML models
# for c in c_array:

#     ensemble_forecast_ML_physical = np.zeros((len(test_sequences), number_of_perturbed_sequences, len(number_timesteps_predict_list), 3))
#     ensemble_forecast_ML_physical_error = np.zeros((len(test_sequences), number_of_perturbed_sequences, len(number_timesteps_predict_list), 3))

#     for num in range(len(test_sequences)):
#         for j, lead_time in enumerate(number_timesteps_predict_list):

#             # Load the models
#             trunctated_filename = f'saved_models/{system}/timesteps_{lead_time}'
#             ML_physical_model = keras.models.load_model(f'{trunctated_filename}/x_transformation_{x_transformation_type}/physical_{system}{x_transformation_type}_c{c}_{number_of_data_points}_{length_of_subsequence}_{lead_time}.keras')
#             ML_physical_error_model = keras.models.load_model(f'{trunctated_filename}/x_transformation_{x_transformation_type}/physical_error_{system}{x_transformation_type}_c{c}_{number_of_data_points}_{length_of_subsequence}_{lead_time}.keras')

#             # Make the ML model predictions
#             physical_input_dictionary = {'input_ob': ensemble_dictionary['perturbed_initial_conditions'][num, :, :, :], 'input_pred': ensemble_dictionary[c]['imperfect_predictions'][num,:,j,:]}
#             ensemble_forecast_ML_physical[num,:,j,:] = ML_physical_model.predict(physical_input_dictionary)
#             ensemble_forecast_ML_physical_error[num,:,j,:] = ML_physical_error_model.predict(physical_input_dictionary)

#     ensemble_dictionary[c]['ML_physical_forecast'] = ensemble_forecast_ML_physical
#     ensemble_dictionary[c]['ML_physical_error_forecast'] = ensemble_forecast_ML_physical_error

# np.save(f'ensemble_dictionaries/ensemble_{system}_lastest_001',ensemble_dictionary)
ensemble_dictionary = np.load(f'ensemble_dictionaries/ensemble_{system}_lastest_001.npy',allow_pickle=True).item()

# Evaluate skill scores
score_list = [proper_linear_score]
score_name_list = ['proper_linear']
for i in range(len(score_list)):
    score = score_list[i]
    score_name = score_name_list[i]
    ensemble_dictionary[f'{score_name}_score'] ={}

    ML_only_scores = np.zeros((len(number_timesteps_predict_list)))
    for j in range(len(number_timesteps_predict_list)):
        ML_only_scores[j] = score(ensemble_dictionary['ML_only_forecast'][:,:,j,:], ensemble_dictionary['observation_answers'][:,j,:])
    ensemble_dictionary[f'{score_name}_score']['ML_only'] = ML_only_scores

    for c in c_array:
        ensemble_dictionary[f'{score_name}_score'][c] = {}
        ML_physical_scores = np.zeros((len(number_timesteps_predict_list)))
        ML_physical_error_scores = np.zeros((len(number_timesteps_predict_list)))
        imperfect_physical_scores = np.zeros((len(number_timesteps_predict_list)))
        for j in range(len(number_timesteps_predict_list)):
            ensemble_dictionary[c]['imperfect_predictions']
            imperfect_physical_scores[j] = score(ensemble_dictionary[c]['imperfect_predictions'][:,:,j,:], ensemble_dictionary['observation_answers'][:,j,:])
            ML_physical_scores[j] = score(ensemble_dictionary[c]['ML_physical_forecast'][:,:,j,:], ensemble_dictionary['observation_answers'][:,j,:])
            ML_physical_error_scores[j] = score(ensemble_dictionary[c]['ML_physical_error_forecast'][:,:,j,:] + ensemble_dictionary[c]['imperfect_predictions'][:,:,j,:], ensemble_dictionary['observation_answers'][:,j,:])
        ensemble_dictionary[f'{score_name}_score'][c]['imperfect_physical'] = imperfect_physical_scores
        ensemble_dictionary[f'{score_name}_score'][c]['ML_physical'] = ML_physical_scores
        ensemble_dictionary[f'{score_name}_score'][c]['ML_physical_error'] = ML_physical_error_scores

np.save(f'ensemble_dictionaries/ensemble_{score_name_list[0]}_{system}_001_latest',ensemble_dictionary)