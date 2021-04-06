import pandas

df = pandas.read_csv('./turnstile_data_master_with_weather.csv')
daily = df['ENTRIESn_hourly'].sum() / 30

print 'Total ridership: ', df['ENTRIESn_hourly'].sum()
print 'Average per day: ', daily
print 'Avergage per week: ', daily  * 7
