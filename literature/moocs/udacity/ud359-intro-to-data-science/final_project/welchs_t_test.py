import pandas
import scipy.stats

data = pandas.read_csv('./turnstile_data_master_with_weather.csv')
clear = data[data.rain == 0]['ENTRIESn_hourly']
wet = data[data.rain == 1]['ENTRIESn_hourly']

print scipy.stats.ttest_ind(wet, clear, equal_var=False)
