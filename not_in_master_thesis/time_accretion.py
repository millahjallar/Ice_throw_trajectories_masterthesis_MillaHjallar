import xarray as xr

weather_path = '/Users/millaregineantonsenhjallar/Library/CloudStorage/OneDrive-UiTOffice365/mesotimeseries-Point 1.nc'
weather_data = xr.open_dataset(weather_path)
weather_df = weather_data.to_dataframe()

start_date = '2013-01-01 00:00:00'
end_date = '2016-12-31 23:00:00'

time = weather_df.index.get_level_values('time')

weather_df = weather_df[(time >= start_date) & (time <= end_date)]
all_time = weather_df.index.get_level_values('time').values

accretion = weather_df[weather_df['ACCRE_CYL'] > 0]
accretion_time = accretion.index.get_level_values('time').values

accretion_time = (len(accretion_time) / len(all_time)) * 100
print(f'Percentage of accretion time: {accretion_time}%')