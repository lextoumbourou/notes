---
title: Lesson 2 - Data Wrangling
date: 2014-04-16 00:00
status: draft
---

# Lesson 2 - Data Wrangling

* Data wrangling overview
    * Converting data into a better format for analysing
    * Data wrangling can be up to 70% of ds's time
* Analysing messy data
    * Important to understand the structure of the data
* Common Data Formats
    * csv
    * xml
    * json
* pandas syntax overview

```
import pandas

baseball_data = pandas.read_csv('file.csv')
print baseball_data['nameFirst']
baseball_data['height_plus_weight'] = baseball_data['height'] + baseball_data['weight']
baseball_data.to_csv('new_file.csv')
```

* Relational databases
    * Database schema - ya'll know the drill
* Simple queries
    * ```SELECT * FROM mytable LIMIT 20```
* To do: look into module ```pandasql```
* GROUP BY AND other aggregate functions
    * Can have multiple GROUP BY clauses
* APIs
    * REST - representational state transfer
* With pandas you can run: ```dataframe.describe()``` to get a quick snapshot of the data (mean, std, min etc)
* Dealing with missing data
    * list wise deletion
        * for example, if one out of two data elements were missing for a person, we would exclude both from any analysis
    * pairwise deletion
        * we would use *any* the valid data points for each person in the sample
    * imputation
        * making up for missing data in set
* Imputation using linear regression
    * "Create an equation that predicts missing values using data we have"
    * Eg, put mean into missing data
    * May over or under emphasis certain values and trends
* Example of imputation with pandas

```
from pandas import *
import numpy

baseball = pandas.read_csv(filename)
baseball['weight'] = baseball['weight'].fillna(baseball['weight'].mean())
```

## Opinions

* During the projects, it would be nice to see the way the instructor would do it, after you get the answer right - something to compare your solution against.
