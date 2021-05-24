import scipy.stats
import pandas

data = pandas.read_csv('./turnstile_data_master_with_weather.csv')

with_rain = data[data.rain == 1]['ENTRIESn_hourly']
without_rain = data[data.rain == 0]['ENTRIESn_hourly']
U, p = scipy.stats.mannwhitneyu(with_rain, without_rain)

print U, (p * 2)
