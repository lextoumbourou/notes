---
title: Lesson 3 - Data Analysis
date: 2014-04-16 00:00
status: draft
---

# Lesson 3 - Data Analysis

* Statistical Rigor
    * Significance test
        * How confident are we that a sample of data can prove or disprove an assumption?
    * "Formalized framework for comparing and evaluating data"
* Running statistical significance tests
    * Many test make assumptions about data's distribution
* Normal Distributions
    * Two parameters
        * Mean (mu)
        * Std (sigma)
    * Variance = sigma^2

<img src="./images/normal-distribution.png"></img>

* null hypothesis
    * statement to 'disprove' or reject with a test
* Welch's T-test
    * Used for comparing two samples which don't necessarily have the same sample size
    * Formula:

    <img src="./images/welch-formula.png"></img>

    * In code:

    ```
    import math

    def welch_t(mu_1, mu_2, var_1, var_2, n1, n2):
        difference = (mu - mu_2)
        return difference / math.sqrt(
            (var_1 / n1) + (var_2 / n2))
    ```

    * Calculate degrees of freedom (aka nu)

    <img src="./images/calculate-degrees-of-freedom.png"></img>

    * In code:

    ```
    import math

    def welch_df(mu_1, mu_2, var_1, var_2, n1, n2):
        var_with_n_factored = (var_1 / n1 + var_2 / n2) ** 2 
        return (var_with_n_factored / 
            (var_1*2) / n1**2*(n-1) + (var_2*2) / n2**2*(n-1)
        )
    ```

    * Once you have t and nu, you can calculate p-value
        * "p == probability of obtaining the t-statistic as extreme as the one observed, if null was true"
* ttests in Python
    * Use function ```scipy.stats.ttest_ind```
    * With args: ```(list_1, list_2, equal_var=False)```
        * the equal_var=False means it uses Welch's formula
    * Returns tuple ```(t_stat, two_sided_p_value)```
    * To get one-sided p_value, divide by half and ensure greater or less than 0 (depending on whether you're looking for positive or negative results)
* Pandas refresher
    * read dataframe from CSV

    ```
    import pandas
    df = pandas.read_csv('~/filename')
    ```

    * filter dataframe

    ```
    middleweights = df[df.weightclass == 'middle']
    ```

* Non-parametric tests
    * a test that doesn't assume data is drawn from any underlying prob distribution
    * Mann-Whitney U Test
        * Tests null hypothesis that two populations are the same

        ```
        u, p = scipy.stats.mannwhitneyu(x, y)
        ```

* Non-normal data
    * Shapiro-wilk test
        * ```w_test_stat, prob = scipy.stats.shapiro(data)```
* Machine learning
    * "Branch of AI that focuses on constructing systems from large amounts of data to make predictions"
    * Differs from stats because it's focused on making predictions rather than drawing conclusions
    * Types of ML problems
        * Supervised learning
            * Spam filter
        * Unsupervised learning
            * Split photos into groups without tell it what groups to use
* Prediction with Regression
    * Takes in data points as input variables and build most accurate equation
* Linear Regression with Gradient Descent
    * cost function: want to minimize J(theta)
* How to minimize cost function
    1. Start with some Theta value
    2. For each Theta, update Theta values according to this equation

    <img src="./images/theta-equation.png"></img>

        theta - alpha / m * numpy.dot((predicted_values) - values), features)
* Need to understand what the hell ```numpy.dot``` does. It's never explained.
* Coefficients of determination (R-squared)
    1. Have a number of data values ```y[i] through to y[n]```
    2. Have a bunch of predictions ```f[i] through to f[n]```
    3. Average value for the data ```y_bar```
    * R^2 is:

    ```
    1 - (
      sum(y[i] - f[i])^2) / sum((y[i] - y_bar)^2)
    )
    ```

    * Point of it is to determine how effective your coefficients you've gotten from linear regression
* Additional considerations
    * Other types of linear regression
        * Ordinary least squares regression
            * Guaranteed to find optimal solution when performing linear regression
    * Parameter estimation
        * Putting confidence intervals on parameters
    * Underfitting data
        * Trying to fit linear model to non-linear data
    * Gradient descent might find a "local minimum" that's not the actual minimum
        * Could random start points and compare results
* Kurt's Advice for Machine Learning practises
    * With any problem:
        * What do we know?
        * What expectations do we have?
        * Is there any intuition for the data?
    * Pick which part of data science you are most interested in and focus on it
        * Coding
        * Maths/stats
        * Communication
