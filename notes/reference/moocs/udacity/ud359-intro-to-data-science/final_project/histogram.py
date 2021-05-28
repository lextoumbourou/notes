import numpy as np
import pandas
import matplotlib.pyplot as plt

data = pandas.read_csv('./turnstile_data_master_with_weather.csv')
plt.figure()
data[data.rain == 1]['ENTRIESn_hourly'].hist()
data[data.rain == 0]['ENTRIESn_hourly'].hist()
print plt
