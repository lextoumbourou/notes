import pandas
import scipy.stats

data = pandas.read_csv('./turnstile_data_master_with_weather.csv')

print "Shapiro-Wilk rain: "
print scipy.stats.shapiro(data[data.rain == 1]['ENTRIESn_hourly'])
print "Shapiro-Wilk dry: "
print scipy.stats.shapiro(data[data.rain == 0]['ENTRIESn_hourly'])
