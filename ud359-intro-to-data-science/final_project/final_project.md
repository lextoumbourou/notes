# MTA Subway Network question: how does rain affect ridership?

The MTA (Metropolitan Transportation Authority) subway network runs from New York City through Long Island, southeastern New York State, and Connecticut. It serves on average over 8 million people per week [0](http://web.mta.info/mta/network.htm) and has over 8 thousand rail and subway cars[0]. It has been featured in countless Hollywood movies and is a quintessential part of New York culture. New York as a state has fairly interesting weather patterns, it fluxutaes from extremely hot (over ?? celcius) and very cold. 

In this short article, we're going to do some analysis on how weather affects the MTAs ridership, asking the question: what happens when it rains?

## Dataset

The dataset includes data from roughly 550 stations across New York. It begins on the 1st of May and ends on the 30th. 

We could begin our analysis by plotting the determining mean ridership by hour for the month of May for our entire dataset.

<img src="../images/entries-per-hour.png"></img>

(The code used to generate the line chart can be found [here](./entries_per_hour.py))

So, we can see from the graph, that the business hour for the station is around ?? o'clock.

Now let's create the same line graph, but this time we'll separate wet days with non-wet days. The rainy days will be displayed in ??

<img src="../images/entries-per-hour-wet-to-dry.png"></img>

It appears, based on this chart alone, that there is a slight increase in ridership on wetter days?

But it's hard to say. Enter: statistical rigour.

Firstly, let's perform some basic statistical comparisons between the two datasets. [code here](./compare_datasets.py)

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

Interesting. So, let's take a look at our hypothesis.

Our null hypothesis:

null = rain == fog

alternative = rain != fog

When we compare the two samples, we note that the variance differs, as does the sample size. Therefore, Welch's t-test may be an appropriate test to determine whether the sample difference is statistically significant.

In this example, we're setting a p-critical value of 0.05. We'll perform a two-tailed test, with the following result: 0.7861709004186308

Based on this result, we fail to reject the null hypothesis.

Part of the issue with this data set is how small the sample size is for the fog data, in particular.

## Predicting Ridership Per Station

Using Linear Regression, we can determine a set of theta values that can be used to take the dot product of the following input variables. The model we'll use for this is Gradient Descent.

If we take in the following variables: 'rain', 'precipi', 'Hour', 'meantempi', we utilise it to perform Gradient Descent. Setting a learning rate of 0.5 and 50 as the number of iterations, we get the following Theta results for each of the features, respectively: 3.57746093e+00   1.12934079e+01   2.04990276e+02  -2.66371483e+01

Why would provide us with an r squared is 0.45804446474. Not exactly ideal but someway toward working through an accuractely prediction.

## MapReduce on larger dataset

If we wanted to perform some similar calculations for a larger dataset, we could look at introduction MapReduce to perform some calculcations. 

Perhaps we wanted to write some MapReduce code to determine top 10 stations for weekly average ridership?

We could put the large dataset onto a HDFS, then we might write the following [mapper](./mapper.py) and [reducer](./reducer.py).

Which, using the current dataset, returns the following information.

[('R018', 1389878.0), ('R012', 1564752.0), ('R011', 1582914.0), ('R055', 1607534.0), ('R179', 1618261.0), ('R046', 1695150.0), ('R033', 1711663.0), ('R022', 1796932.0), ('R084', 1809423.0), ('R170', 2887918.0)]

Where R170 appears to be the busiest station.

## Conclusion

Based on the data collection, we can infer that rain has a slight correlation with higher ridership on the MTA subway system.

