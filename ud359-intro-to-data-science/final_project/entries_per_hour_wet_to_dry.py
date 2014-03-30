import pandas
import vincent
vincent.core.initialize_notebook()

t_data = pandas.read_csv('./turnstile_data_master_with_weather.csv')

rain = t_data[t_data.rain == 1].groupby('Hour')['ENTRIESn_hourly'].mean()
clear = t_data[t_data.rain == 0].groupby('Hour')['ENTRIESn_hourly'].mean()
data = pandas.DataFrame({'rain': rain, 'clear': clear, 'Hour': rain.index})

line = vincent.Line(data)
line.axis_titles(x='Hour', y='Mean entities per hour')
vincent.core.initialize_notebook()
line.display()
