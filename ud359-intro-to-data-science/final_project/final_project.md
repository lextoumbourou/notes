# MTA Subway Network question: how does rain affect ridership?

## Overview

The MTA (Metropolitan Transportation Authority) subway network runs from New York City through Long Island, southeastern New York State, and Connecticut.

The subway is a quintessential piece of New York culture; having been featured in countless novels, movies and TV shows.

For subway operators, predicting and understanding the connection between people's behaviour and weather conditions could be extremely useful for operational planning. So, in this short article, we're going to do some analysis on how weather affects the MTAs ridership, specifically asking the question: what happens when it rains?

## Dataset

Our sample dataset includes data from roughly 550 stations across New York. It begins on the 1st of May and ends on the 30th. 

According to our dataset, there were over 144,532,327 people entering the turnstile for the month, an average of about 4,817,744 per day and about 33,724,209 entries per week.

([basic_stats.py](https://github.com/lextoumbourou/study-notes/blob/master/ud359-intro-to-data-science/final_project/basic_stats.py)).

We begin our analysis by plotting the mean ridership by hour for the month of May across our entire dataset.

### Mean Ridership Per Hour

<img src="https://raw.githubusercontent.com/lextoumbourou/study-notes/master/ud359-intro-to-data-science/final_project/images/mean-entries-per-hour.png"></img>

[entries_per_hour.py](https://github.com/lextoumbourou/study-notes/blob/master/ud359-intro-to-data-science/final_project/entries_per_hour.py)

From simply examining the graph, we can see that the busiest hour for the station is around 20:00.

Now let's create the same line graph, but this time we'll separate wet days with non-wet days.

### Mean Ridership Per Hour (Wet vs Dry days)

<img src="https://raw.githubusercontent.com/lextoumbourou/study-notes/master/ud359-intro-to-data-science/final_project/images/mean-entries-per-hour-wet-vs-dry.png"></img>

[entries_per_hour_wet_to_dry.py](https://github.com/lextoumbourou/study-notes/blob/master/ud359-intro-to-data-science/final_project/entries_per_hour_wet_to_dry.py)

It appears, based on this chart alone, that there is a slight increase in ridership on wetter days.

However, more statistical rigour is required before coming to any correlation conclusions.

## Statistical Analysis

To perform a statistical comparison of the two samples group, we first need to understand the distribution of the data. By plotting the hourly ridership for both sample groups, we get a reasonable sense that the data is not normally distributed.

### Histogram of Ridership Per Hour

<img src="https://raw.githubusercontent.com/lextoumbourou/study-notes/master/ud359-intro-to-data-science/final_project/images/histogram-entries.png"></img>

[histogram.py](https://github.com/lextoumbourou/study-notes/blob/master/ud359-intro-to-data-science/final_project/histogram.py)

However, we can also perform a Shapiro-Wilk test, a statistical test of distribution normality, to help us determine that more definitively. Running it against our data, we're returned a p-value for our test statistic of 0.0, which strongly suggest we can reject the null hypothesis that our data is from a normal distribution. 

[shapiro_wilk.py](https://github.com/lextoumbourou/study-notes/blob/master/ud359-intro-to-data-science/final_project/shapiro_wilk.py)

Therefore, the non-parametric Mann Whitney U Test will be an appropriate test to compare the samples, since it is known to have greater efficiency than a t-test on non-normal distributions.

In this example, we're setting a p-critical value of 0.05. We'll perform a two-tailed test, with the following U-value: 1924409167.0 and a two-tailed p-value: 0.0386192688276. Based on this, we reject the null hypothesis. It appears rain does affect ridership. However, more data would be required for clarity of these results.

[mann_whitney_i.py](https://github.com/lextoumbourou/study-notes/blob/master/ud359-intro-to-data-science/final_project/mann_whitney_i.py)

## Predicting Ridership Per Station

We can also use Linear Regression to help predict ridership at a station based on some set of parameters. Linear Regression's goal is to effectively draw a "best fit" line through historic data, which can then be used to predict values into the future. It works by taking in a set of input values, then for each input value seeking to find a coefficient, or Theta value, which can be multipled to create a cumulative output variable. The Theta values, therefore, provide an indication of the importance of each input variable in predicting the output variable.

There are a number of algorithms one can use for Linear Regression. In our model, we'll use Gradient Descent. Gradient Descent requires a Cost Function which provides a measure of how accurate the predicted values are compared to actual values. Then, it utilises a search algorithm which seeks to find the set of Theta values that can minimize the Cost Function. A learning rate, or alpha, needs to be set in order to determine how "fast" the algorithm will iterate through the search. A learning rate too small could result in the algorithm taking an excessively long time to converge. A rate too high, could result in the algorithm missing the minima of the Cost Function.

In our model, we're using rain, precipi, Hour, meantempi and station as input values. We set a learning rate of 0.5 - a happy medium between learning too fast and over utilising our computational resources - and the value 50 as the maximum number of iterations. This returns us an r-squared value of 0.45804446474, not perfect but moves us some way toward finding an optimal solution.

Some shortcomings of this approach are that Gradient Descent may not an optimal solution as opposed to other algorithms, like the Ordinary Least Squares regression (OLS), which are guaranteed to find the optimal solution. Additionally, in our example, we are determining hard-coded Theta values for our input variables. However, in the real world, the Theta values might have some confidence intervals allowing us to answer questions like: how likely is it we'd get a Theta value for this input variable if the variable had no effect on the output variable? Lastly, we should consider the possibility of our algorithm getting stuck at a Cost Function's "local minima", where a lower value could be found by taking a different path through the data. Such an example could be mitigated by running the algorithm numerous times using randomised initial Theta values and contrasting the results.

[gradient_descent.py](https://github.com/lextoumbourou/study-notes/blob/master/ud359-intro-to-data-science/final_project/gradient_descent.py)

## Working with a larger dataset 

If we wanted to perform some similar calculations for a larger dataset, we could look at introduction MapReduce to perform some calculations. 

For example, we could write MapReduce code to determine top 10 stations for weekly average ridership. To do that, we'd first put the large dataset onto a HDFS. Then we might write a mapper and reducer like the following: [mapper.py](https://github.com/lextoumbourou/study-notes/blob/master/ud359-intro-to-data-science/final_project/mapper.py) and [reducer.py](https://github.com/lextoumbourou/study-notes/blob/master/ud359-intro-to-data-science/final_project/reducer.py).

Which, using the current dataset, returns the following information.

```
('R170', 2887918.0)
('R084', 1809423.0)
('R022', 1796932.0)
('R033', 1711663.0)
('R046', 1695150.0)
('R179', 1618261.0)
('R055', 1607534.0)
('R011', 1582914.0)
('R012', 1564752.0)
('R018', 1389878.0)
```

Based on this, it appears R170 is the busiest turnstile.

## Conclusion

Based on the tools available to us, we can infer that rain does have a statistical correlation with ridership on the MTA subway system. With that said, one would expect with more data we could get closer to a more definitive answer. Using Gradient Descent, we can predict ridership using input variables relating to the weather, providing us with a moderately accurate prediction model. Lastly, with a larger data size, map reduce techniques could help us scale out our processing to help answer additional questions about the data.
