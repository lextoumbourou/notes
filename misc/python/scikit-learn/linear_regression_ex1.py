"""
Simple Linear Regression example using scikit-learn.

Source: http://bigdataexaminer.com/uncategorized/how-to-run-linear-regression-in-python-scikit-learn/
"""

import numpy
import pandas
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets, cross_validation
from sklearn.linear_model import LinearRegression


if __name__ == '__main__':
    # Load Dataset and create DataFrame
    boston = datasets.load_boston()
    boston_df = pandas.DataFrame(boston.data)
    boston_df.columns = boston.feature_names

    # Create Linear Regression model
    lm = LinearRegression()

    # Fit the training data
    lm.fit(boston_df, boston.target)

    print
    print 'Number of coefficients: {0}'.format(len(lm.coef_))
    print 'Estimated intercept coefficient: {0}'.format(lm.intercept_)
    print

    # Print Dataframe containing features and estimated coefficients
    print pandas.DataFrame(
        zip(boston_df.columns, lm.coef_),
        columns=['features', 'estimated coefficients'])

    # Create a scatter plot comparing average rooms per dwelling to price
    plt.scatter(boston_df.RM, boston.target)
    plt.xlabel('Average number of rooms per dwelling (RM)')
    plt.ylabel('Housing price')
    plt.title('Relationship between RM and price')
    plt.show()

    # Predict house prices using features used for training
    plt.scatter(boston.target, lm.predict(boston_df))
    plt.xlabel('Prices: $Y_i$')
    plt.ylabel('Predicted prices: $\hat{Y}_i$')
    plt.title('Prices vs Predicted Prices: $Y_i$ vs $\hat{Y}_i$')

    # Calculate mean squared error
    mse = numpy.mean((boston.target - lm.predict(boston_df)) ** 2)
    print 'Mean squared error of predictions with all features: {}'.format(mse)

    # Split data into training and test datasets
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(
        boston_df, boston.target, test_size=0.33, random_state=5)
    lm = LinearRegression()
    lm.fit(x_train, y_train)
    pred_test = lm.predict(x_test)
    mse = numpy.mean((y_test - pred_test) ** 2)
    print 'Mean squared error of predictions with test split: {}'.format(mse)
