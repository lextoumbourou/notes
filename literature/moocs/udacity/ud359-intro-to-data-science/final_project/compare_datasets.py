import pandas

t_data = pandas.read_csv('./turnstile_data_master_with_weather.csv')
rain_daily_total = t_data[t_data.rain == 1].groupby('DATEn')['ENTRIESn_hourly'].sum()
clear_daily_total = t_data[t_data.rain == 0].groupby('DATEn')['ENTRIESn_hourly'].sum()
pandas.DataFrame({'rain': rain_daily_total, 'clear': clear_daily_total}).describe()
