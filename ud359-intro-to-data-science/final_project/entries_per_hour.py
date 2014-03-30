import pandas
import vincent
vincent.core.initialize_notebook()

t_data = pandas.read_csv('./turnstile_data_master_with_weather.csv')

data = t_data.groupby('Hour')['ENTRIESn_hourly'].mean()
line = vincent.Line(data)
line.axis_titles(x='Hour', y='Mean entities per hour')
vincent.core.initialize_notebook()
line.display()
