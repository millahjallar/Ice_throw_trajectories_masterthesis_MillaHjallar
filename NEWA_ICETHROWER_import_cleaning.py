import numpy as np
np.random.seed(12345)
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from windrose import WindroseAxes
import matplotlib.ticker as mtick
import matplotlib.colors as colors

N = 100000
RHO_ICE = 520.0                   # Density of ice (Snow ice blend) [kg/m^3]

'''
wind_rose                        : Set to False if you don't want to create wind rose plot.
wind_rose_accretion_only         : Set to False if you don't want to create wind rose plot with only accretion hours.
wind_distribution                : Set to False if you don't want to create wind distribution plot.
wind_distribution_accretion_only : Set to False if you don't want to create wind distribution plot with only accretion hours.
'''

wind_rose = False
wind_rose_accretion_only = False
wind_distribution = False
wind_distribution_accretion_only = False

weather_path = '/Users/millaregineantonsenhjallar/Library/CloudStorage/OneDrive-UiTOffice365/mesotimeseries-Point 1.nc'
weather_data = xr.open_dataset(weather_path)
weather_df = weather_data.to_dataframe()

start_date = '2013-01-01 00:00:00'
end_date = '2016-12-31 23:00:00'

time = weather_df.index.get_level_values('time')

weather_data_time = weather_df[(time >= start_date) & (time <= end_date)]
weather_df = weather_data_time[weather_data_time['ACCRE_CYL'] > 0]

wspd_time = weather_data_time['WS'].values
wdir_time = weather_data_time['WD'].values
time_time = weather_data_time.index.get_level_values('time')

if wind_distribution:
    plt.figure(figsize=(10, 6))
    plt.hist(wspd_time, bins=80, edgecolor='white')
    plt.xlabel('Wind speed [m/s]')
    plt.ylabel('Frequency')
    plt.title(f'Wind speed distribution with and without accretion hours')
    plt.text(0.95, 0.95, f'Mean wind speed: {np.mean(wspd_time):.2f} m/s', horizontalalignment='right',
             verticalalignment='top', transform=plt.gca().transAxes)
    plt.grid(True)
    plt.show()
    print(max(wspd_time))

if wind_rose:
    def truncate_colormap(color, minval=0.0, maxval=1.0, n=100):
        cmap = plt.colormaps.get_cmap(color)
        new_cmap = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap

    cmap = truncate_colormap('Blues', 0.1, 0.9)
    #cmap = truncate_colormap('GnBu', 0.1, 0.9)

    ax = WindroseAxes.from_ax()
    ax.bar(wdir_time, wspd_time, normed=True, cmap=cmap, opening=0.8, edgecolor='white')
    ax.set_legend()
    #ax.set_title(f'Windrose with and without accretion hours')
    fmt = '%.0f%%'
    yticks = mtick.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(yticks)
    plt.show()

wspd = weather_df['WS'].values
wdir = weather_df['WD'].values
ice_acc = weather_df['ACCRE_CYL'].values
time = weather_df.index.get_level_values('time')
temperature = weather_df['T'].values

if wind_distribution_accretion_only:
    plt.figure(figsize=(10, 6))
    plt.hist(wspd, bins=80, edgecolor='white')
    plt.xlabel('Wind speed [m/s]')
    plt.ylabel('Frequency')
    plt.title(f'Wind speed distribution with only accretion hours')
    plt.text(0.95, 0.95, f'Mean wind speed: {np.mean(wspd):.2f} m/s', horizontalalignment='right',
             verticalalignment='top', transform=plt.gca().transAxes)
    plt.grid(True)
    plt.show()
    print(max(wspd))

if wind_rose_accretion_only:
    def truncate_colormap(color, minval=0.0, maxval=1.0, n=100):
        cmap = plt.colormaps.get_cmap(color)
        new_cmap = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap

    cmap = truncate_colormap('Blues', 0.1, 0.9)
    #cmap = truncate_colormap('GnBu', 0.1, 0.9)

    ax = WindroseAxes.from_ax()
    ax.bar(wdir, wspd, normed=True, cmap=cmap, opening=0.8, edgecolor='white')
    ax.set_legend()
    #ax.set_title(f'Windrose with only accretion hours')
    fmt = '%.0f%%'
    yticks = mtick.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(yticks)
    plt.show()

'''# converting from kelvin to celsius
temps = temperature - 273.15

pos = temps >  0
neg = temps <  0
zer = temps == 0

plt.figure(figsize=(17, 7))
plt.plot(time[pos], temps[pos], color='red',    alpha=0.5, label='> 0 °C')
plt.plot(time[neg], temps[neg], color='blue',   alpha=0.5, label='< 0 °C')
plt.plot(time[zer], temps[zer], color='black',  alpha=0.5, label='= 0 °C')
# make a line along the x-axis at 0 °C
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.xlabel('Time')
plt.ylabel('Temperature [°C]')
plt.title('Temperature without accretion hours')
plt.legend()
plt.show()'''


'''plt.bar(time, temperature, width=10, color='blue', alpha=0.5, label='Ice accretion hours')
plt.show()'''

'''plt.figure(figsize=(17, 7))
plt.plot(time, wspd, label='Wind speed as a function of time', zorder=5)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.xlabel('Time')
plt.ylabel('Wind speed [m/s]')
plt.title('Wind speed without accretion hours')
plt.show()'''

'''print('Mean wind speed with and without accretion hours:', np.mean(wspd_time))
print('Mean wind speed with only accretion hours:', np.mean(wspd))
print('Max wind speed with and without accretion hours:', np.max(wspd_time))
print('Max wind speed with only accretion hours:', np.max(wspd))
print('Mean wind direction with and without accretion hours:', np.mean(wdir_time))
print('Mean wind direction with only accretion hours:', np.mean(wdir))'''

# ------------------------------------------------------------------------------------------
# ---------------------- Importing necessary data for area and masses ----------------------
# ------------------------------------------------------------------------------------------

file_path = 'IceThrower_dataBase_2013_2016.xlsx'

info_data = pd.read_excel(file_path, sheet_name='Info', skiprows=4)
dv_data = pd.read_excel(file_path, sheet_name='DV', skiprows=4, decimal=',')
vf1_data = pd.read_excel(file_path, sheet_name='VF 1', skiprows=4, decimal=',')
vf2_data = pd.read_excel(file_path, sheet_name='VF 2', skiprows=4, decimal=',')
vf3_data = pd.read_excel(file_path, sheet_name='VF 3', skiprows=4, decimal=',')
vf4_data = pd.read_excel(file_path, sheet_name='VF 4', skiprows=4, decimal=',')
vf5_data = pd.read_excel(file_path, sheet_name='VF 5', skiprows=4, decimal=',')
vf6_data = pd.read_excel(file_path, sheet_name='VF 6', skiprows=4, decimal=',')
SK_data = pd.read_excel(file_path, sheet_name='SK', skiprows=4, decimal=',')

# ---------------------- Cleaning datasets ----------------------

# Removing NaNs / 'N' for MASS
dv_data = dv_data[(dv_data['Mass              (kg)'] != 'N') & (dv_data['Mass              (kg)'].notna())]
vf1_data = vf1_data[(vf1_data['Mass (kg)'] != 'N') & (vf1_data['Mass (kg)'].notna())]
vf2_data = vf2_data[(vf2_data['Mass            (kg)'] != 'N') & (vf2_data['Mass            (kg)'].notna())]
vf3_data = vf3_data[(vf3_data['Mass (kg)'] != 'N') & (vf3_data['Mass (kg)'].notna())]
vf4_data = vf4_data[(vf4_data['Mass (kg)'] != 'N') & (vf4_data['Mass (kg)'].notna())]
vf5_data = vf5_data[(vf5_data['Mass (kg)'] != 'N') & (vf5_data['Mass (kg)'].notna())]
vf6_data = vf6_data[(vf6_data['Mass (kg)'] != 'N') & (vf6_data['Mass (kg)'].notna())]
SK_data  = SK_data[ (SK_data['Mass (kg)'] != 'N')  & (SK_data['Mass (kg)'].notna())]

# Removing NaNs / 'N' for LENGTH
dv_data = dv_data[(dv_data['Length (cm)'] != 'N') & (dv_data['Length (cm)'].notna())]
vf1_data = vf1_data[(vf1_data['Lenght           (cm)'] != 'N') & (vf1_data['Lenght           (cm)'].notna())]
vf2_data = vf2_data[(vf2_data['Length (cm)'] != 'N') & (vf2_data['Length (cm)'].notna())]
vf3_data = vf3_data[(vf3_data['Lenght (cm)'] != 'N') & (vf3_data['Lenght (cm)'].notna())]
vf4_data = vf4_data[(vf4_data['Lenght (cm)'] != 'N') & (vf4_data['Lenght (cm)'].notna())]
vf5_data = vf5_data[(vf5_data['Length (cm)'] != 'N') & (vf5_data['Length (cm)'].notna())]
vf6_data = vf6_data[(vf6_data['Length (cm)'] != 'N') & (vf6_data['Length (cm)'].notna())]
SK_data  = SK_data[ (SK_data['Length (cm)'] != 'N') & (SK_data['Length (cm)'].notna())]

# Removing NaNs / 'N' for WIDTH
dv_data = dv_data[(dv_data['Width         (cm)'] != 'N') & (dv_data['Width         (cm)'].notna())]
vf1_data = vf1_data[(vf1_data['Width               (cm)'] != 'N') & (vf1_data['Width               (cm)'].notna())]
vf2_data = vf2_data[(vf2_data['Width (cm)'] != 'N') & (vf2_data['Width (cm)'].notna())]
vf3_data = vf3_data[(vf3_data['Width (cm)'] != 'N') & (vf3_data['Width (cm)'].notna())]
vf4_data = vf4_data[(vf4_data['Width (cm)'] != 'N') & (vf4_data['Width (cm)'].notna())]
vf5_data = vf5_data[(vf5_data['Width (cm)'] != 'N') & (vf5_data['Width (cm)'].notna())]
vf6_data = vf6_data[(vf6_data['Width (cm)'] != 'N') & (vf6_data['Width (cm)'].notna())]
SK_data  = SK_data[ (SK_data['Width (cm)'] != 'N')  & (SK_data['Width (cm)'].notna())]


# Removing outliers
rem_mass1 = vf2_data[vf2_data['Mass            (kg)'] == 0.7].index
vf2_data = vf2_data.drop(rem_mass1)

# ---------------------- Defining variables/constants ----------------------

# Mass
dv_mass  = pd.to_numeric(dv_data.loc[:,  'Mass              (kg)'])
vf1_mass = pd.to_numeric(vf1_data.loc[:, 'Mass (kg)'])
vf2_mass = pd.to_numeric(vf2_data.loc[:, 'Mass            (kg)'])
vf3_mass = pd.to_numeric(vf3_data.loc[:, 'Mass (kg)'])
vf4_mass = pd.to_numeric(vf4_data.loc[:, 'Mass (kg)'])
vf5_mass = pd.to_numeric(vf5_data.loc[:, 'Mass (kg)'])
vf6_mass = pd.to_numeric(vf6_data.loc[:, 'Mass (kg)'])
SK_mass  = pd.to_numeric(SK_data.loc[:,   'Mass (kg)'])

# Length
dv_lenght  = pd.to_numeric(dv_data.loc[:,  'Length (cm)'])
vf1_lenght = pd.to_numeric(vf1_data.loc[:, 'Lenght           (cm)'])
vf2_lenght = pd.to_numeric(vf2_data.loc[:, 'Length (cm)'])
vf3_lenght = pd.to_numeric(vf3_data.loc[:, 'Lenght (cm)'])
vf4_lenght = pd.to_numeric(vf4_data.loc[:, 'Lenght (cm)'])
vf5_lenght = pd.to_numeric(vf5_data.loc[:, 'Length (cm)'])
vf6_lenght = pd.to_numeric(vf6_data.loc[:, 'Length (cm)'])
SK_lenght  = pd.to_numeric(SK_data.loc[:,   'Length (cm)'])

# Width
dv_width  = pd.to_numeric(dv_data.loc[:,  'Width         (cm)'])
vf1_width = pd.to_numeric(vf1_data.loc[:, 'Width               (cm)'])
vf2_width = pd.to_numeric(vf2_data.loc[:, 'Width (cm)'])
vf3_width = pd.to_numeric(vf3_data.loc[:, 'Width (cm)'])
vf4_width = pd.to_numeric(vf4_data.loc[:, 'Width (cm)'])
vf5_width = pd.to_numeric(vf5_data.loc[:, 'Width (cm)'])
vf6_width = pd.to_numeric(vf6_data.loc[:, 'Width (cm)'])
SK_width  = pd.to_numeric(SK_data.loc[:,   'Width (cm)'])

masses0 = np.concatenate((dv_mass, vf1_mass, vf2_mass, vf3_mass, vf4_mass, vf5_mass, vf6_mass, SK_mass))
lengths0 = np.concatenate((dv_lenght, vf1_lenght, vf2_lenght, vf3_lenght, vf4_lenght, vf5_lenght, vf6_lenght, SK_lenght))
widths0 = np.concatenate((dv_width, vf1_width, vf2_width, vf3_width, vf4_width, vf5_width, vf6_width, SK_width))

areas = []
masses = []
#lengths = []
#widths = []

def thickness_from_mass(mass, length, width, rho_ice):
    return mass / (length * width * rho_ice)

def average_projected_area(length, width, thickness):
    # Simple average of the 3 orthographic projections
    return (length * width + width * thickness + length * thickness) / 3

for i in range(len(lengths0)):
    L_m = lengths0[i] / 100.0
    W_m = widths0[i]  / 100.0
    m   = masses0[i]
    T_m = thickness_from_mass(m, L_m, W_m, RHO_ICE)
    A_eff = average_projected_area(L_m, W_m, T_m)

    #lengths.append(L_m)
    #widths.append(W_m)
    areas.append(A_eff)
    masses.append(m)

masses = np.array(masses)
areas = np.array(areas)

overshoot = 1.5
M = int(N * overshoot)

idx = np.random.choice(len(masses), size=M, replace=True)
mass_tmp = masses[idx] + np.random.normal(0, 0.2 * np.median(masses), size=M)
area_tmp = areas[idx] + np.random.normal(0, 0.2 * np.median(areas), size=M)

mask = (mass_tmp > min(masses)) & (area_tmp > min(areas))
mass_dist = mass_tmp[mask]
area_dist = area_tmp[mask]

mass_distribution = mass_dist[:N]
area_distribution = area_dist[:N]

# idx = np.random.choice(len(masses), size=N, replace=True)
# mass_distribution = masses[idx].copy()
# area_distribution = areas[idx].copy()

# mass_noise_scale = 0.1 * np.median(masses)
# area_noise_scale = 0.1 * np.median(areas)
# mass_distribution += np.random.normal(0, mass_noise_scale, size=N)
# area_distribution += np.random.normal(0, area_noise_scale, size=N)

# mask = (mass_distribution > 0) & (area_distribution > 0)
# mass_distribution = mass_distribution[mask]
# area_distribution = area_distribution[mask]


if __name__ == '__main__':
    plt.figure(figsize=(10, 6))
    plt.hist(masses, bins=80, edgecolor='white')
    plt.xlabel('Mass [kg]')
    plt.ylabel('Frequency')
    plt.title(f'Mass distribution without noise addition')
    plt.text(0.95, 0.95, f'Mean mass: {np.mean(masses):.2f} kg', horizontalalignment='right',
            verticalalignment='top', transform=plt.gca().transAxes)
    plt.grid(True)
    plt.legend()
    plt.show()


    plt.figure(figsize=(10, 6))
    plt.hist(mass_distribution, bins=80, edgecolor='white')
    plt.xlabel('Mass [kg]')
    plt.ylabel('Frequency')
    plt.title(f'Mass distribution with noise addition')
    plt.text(0.95, 0.95, f'Mean mass: {np.mean(mass_distribution):.2f} kg', horizontalalignment='right',
            verticalalignment='top', transform=plt.gca().transAxes)
    plt.grid(True)
    plt.legend()
    plt.show()









'''def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

# Finding array of 6 almost equally spaced masses with corresponding areas
print(min(masses), max(masses))
masses_array = np.linspace(min(masses), max(masses), 6)
nearest_mass_idx = find_nearest(masses, masses_array[4])
# finding out what value that is both for mass and area
nearest_mass = masses[nearest_mass_idx]
nearest_area = areas[nearest_mass_idx]
print(f'5: Nearest mass: {nearest_mass} kg')
print(f'5: Nearest area: {nearest_area} m^2')
quit()'''