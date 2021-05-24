import numpy as np
import pandas

def normalize_features(array):
    mu = array.mean()
    array_normalized = (array-mu)/array.std()
    sigma = array.std()

    return array_normalized, mu, sigma

def compute_cost(features, values, theta):
    m = len(values)
    sum_of_square_errors = np.square(np.dot(features, theta) - values).sum()
    cost = sum_of_square_errors / (2*m)
    
    return cost

def gradient_descent(features, values, theta, alpha, num_iterations):
    m = len(values)
    cost_history = []

    for i in range(num_iterations):
        predicted_values = np.dot(features, theta)
        theta = theta - alpha / m * np.dot((predicted_values - values), features)
        
        cost = compute_cost(features, values, theta)
        cost_history.append(cost)
  
    print "Theta is ", theta
    return theta, pandas.Series(cost_history)

def predictions(dataframe):
    dummy_units = pandas.get_dummies(dataframe['UNIT'], prefix='unit')
    features = dataframe[['rain', 'precipi', 'Hour', 'meantempi']].join(dummy_units)
    values = dataframe[['ENTRIESn_hourly']]
    m = len(values)

    features, mu, sigma = normalize_features(features)

    features['ones'] = np.ones(m)
    features_array = np.array(features)
    values_array = np.array(values).flatten()

    alpha = 0.5
    num_iterations = 50

    #Initialize theta, perform gradient descent
    theta_gradient_descent = np.zeros(len(features.columns))
    theta_gradient_descent, cost_history = gradient_descent(features_array, values_array, theta_gradient_descent,
                                                            alpha, num_iterations)

    predictions = np.dot(features_array, theta_gradient_descent)

    return predictions

def compute_r_squared(data, predictions):
    data_mean = data.mean()
    r_squared = 1 - (
        np.sum(np.square(data - predictions)) / np.sum(np.square(data - data_mean))
    )

    return r_squared
    

if __name__ == '__main__':
    df = pandas.read_csv('./turnstile_data_master_with_weather.csv')
    print "Prediction are ", predictions(df)
    features['ones'] = np.ones(m)
    data = np.array(df['ENTRIESn_hourly']).flatten()

    print "R^2 results ", compute_r_squared(data, predictions)
