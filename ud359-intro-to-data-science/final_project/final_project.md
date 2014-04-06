# MTA Subway Network question: how does rain affect ridership?

## Overview

The MTA (Metropolitan Transportation Authority) subway network runs from New York City through Long Island, southeastern New York State, and Connecticut.

The subway is a quintessential piece of New York culture; having been featued in countless novels, movies and TV shows.

For subway operators, predicting and understanding the connection between people's behaviour and weather conditions can be absolutley essential for operational planning. So, in this short article, we're going to do some analysis on how weather affects the MTAs ridership, specifically asking the question: what happens when it rains?

## Dataset

Our sample dataset includes data from roughly 550 stations across New York. It begins on the 1st of May and ends on the 30th. 

According to our dataset, there were over 144,532,327 people entering the turnstile for the month, an average of about 4,817,744 per day and about 33,724,209 entries per week.

([basic_stats.py](https://github.com/lextoumbourou/study-notes/blob/master/ud359-intro-to-data-science/final_project/basic_stats.py)).

We could begin our analysis by plotting the determining mean ridership by hour for the month of May for the entire dataset.

<img src="https://raw.githubusercontent.com/lextoumbourou/study-notes/master/ud359-intro-to-data-science/final_project/images/mean-entries-per-hour.png"></img>

[entries_per_hour.py](https://github.com/lextoumbourou/study-notes/blob/master/ud359-intro-to-data-science/final_project/entries_per_hour.py)

So, we can see from the graph, that the busiest hour for the station is around 20:00.

Now let's create the same line graph, but this time we'll separate wet days with non-wet days.

<img src="https://raw.githubusercontent.com/lextoumbourou/study-notes/master/ud359-intro-to-data-science/final_project/images/mean-entries-per-hour-wet-vs-dry.png"></img>

[entries_per_hour_wet_to_dry.py](https://github.com/lextoumbourou/study-notes/blob/master/ud359-intro-to-data-science/final_project/entries_per_hour_wet_to_dry.py)

It appears, based on this chart alone, that there is a slight increase in ridership on wetter days.

However, more statistical rigour is required before coming to any correlation conclusions.

## Statistical Anaylsis

Firstly, let's perform some basic statistical comparisons between wet and dry days.

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>clear</th>
      <th>rain</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>      20.000000</td>
      <td>      10.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td> 4788886.000000</td>
      <td> 4875460.700000</td>
    </tr>
    <tr>
      <th>std</th>
      <td> 1392179.929718</td>
      <td> 1336030.366933</td>
    </tr>
    <tr>
      <th>min</th>
      <td> 2370432.000000</td>
      <td> 2661525.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td> 3484968.250000</td>
      <td> 4064350.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td> 5666797.500000</td>
      <td> 5549290.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td> 5864204.500000</td>
      <td> 5806608.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td> 5965262.000000</td>
      <td> 5866031.000000</td>
    </tr>
  </tbody>
</table> 

([compare_datasets.py](https://github.com/lextoumbourou/study-notes/blob/master/ud359-intro-to-data-science/final_project/compare_datasets.py))

Since our sample sizes are so different, Welch's t-test may be an appropriate test to determine whether the sample difference is statistically significant.

In this example, we're setting a p-critical value of 0.05. We'll perform a two-tailed test, with the following t-statistic: 1.1 and a two-tailed p-value: 0.27. Based on this, we fail to reject the null hypothesis. Rain does not appear to affect ridership.

([welchs_t_test.py](https://github.com/lextoumbourou/study-notes/blob/master/ud359-intro-to-data-science/final_project/welchs_t_test.py))

## Predicting Ridership Per Station

Using Gradient Descent, we can collect a set of theta values that can help us predict ridership at a station using the following variables: rain, precipi, Hour and meantempi.

In our model, we set a learning rate of 0.5 - a happy medium between learning too fast and overutilising our computational resources - and the value 50 as the number of iterations. With this, we get the following Theta results for each of the features, respectively: 3.57746093e+00   1.12934079e+01   2.04990276e+02  -2.66371483e+01.

Which would provide us with an r-squared values of 0.45804446474 for our features. Not exactly ideal but moves us someway toward achieving an accuractely prediction.

([gradient_descent.py](https://github.com/lextoumbourou/study-notes/blob/master/ud359-intro-to-data-science/final_project/gradient_descent.py))

## Working with a larger dataset 

If we wanted to perform some similar calculations for a larger dataset, we could look at introduction MapReduce to perform some calculcations. 

Perhaps we wanted to write some MapReduce code to determine top 10 stations for weekly average ridership?

We could put the large dataset onto a HDFS, then we might write a mapper and reducer like the following: [mapper.py](https://github.com/lextoumbourou/study-notes/blob/master/ud359-intro-to-data-science/final_project/mapper.py) and [reducer.py](https://github.com/lextoumbourou/study-notes/blob/master/ud359-intro-to-data-science/final_project/reducer.py).

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

Where R170 appears to be the busiest station.

## Conclusion

Based on the tools available to us, we can infer that rain doesn't have a statical correlation with ridership on the MTA subway system, however. However, one would expect with more data we could get closer to a definitive solution.
