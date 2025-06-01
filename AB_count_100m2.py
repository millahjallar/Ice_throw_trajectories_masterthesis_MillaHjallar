import numpy as np
np.random.seed(12345)
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')

df = pd.read_csv('Ice_thrower_simulation_100000its_20.05.2025.csv')

x_impacts = df['X-coordinate [m]'].to_numpy()
y_impacts = df['Y-coordinate [m]'].to_numpy()

safety_distance = 1.5 * (95 + 90)

bin_size = 10.0 # m
# floor=minimum integer, ceil=maximum integer
x_edges = np.arange(np.floor(np.min(x_impacts)), np.ceil(np.max(x_impacts)), bin_size)
y_edges = np.arange(np.floor(np.min(y_impacts)), np.ceil(np.max(y_impacts)), bin_size)

plt.figure(figsize=(10, 8))
#_, _, _, hist = plt.hist2d(x_impacts, y_impacts, bins=[x_edges, y_edges], cmap='Wistia', cmin=1)
#_, _, _, hist = plt.hist2d(x_impacts, y_impacts, bins=[x_edges, y_edges], cmap='viridis_r', cmin=1)
_, _, _, hist = plt.hist2d(x_impacts, y_impacts, bins=[x_edges, y_edges], cmap='jet', cmin=1)
plt.colorbar(hist, label='Counts of ice strikes per 10m x 10m')
plt.scatter(0, 0, s=5, color='lightgrey', label='Turbine tower')
plt.xlabel('X-coordinate of impact location [m]')
plt.ylabel('Y-coordinate of impact location [m]')
circle = plt.Circle((0, 0), safety_distance, color='red', fill=False, linestyle='-', label='Recommended safety distance')
plt.gca().add_patch(circle)
plt.xlim(-350, 350)
plt.ylim(-350, 350)
plt.title('Impact locations of ice throws in xy-plane with counts per 10m x 10m')
plt.legend(loc='upper right')
plt.show()

# ------------------------------------------------------
# ------ filtering out the points with counts < 5 ------
# ------------------------------------------------------

H, x_e, y_e = np.histogram2d(x_impacts, y_impacts, bins=[x_edges, y_edges])
H_m = np.ma.masked_where(H <= 1, H)

plt.figure(figsize=(10, 8))
mesh = plt.pcolormesh(x_e, y_e, H_m.T, cmap='jet', edgecolors='none')
plt.colorbar(mesh, label='Counts per 10 m  10 m (only counts > 1)')
plt.scatter(0, 0, s=5, color='lightgrey', label='Turbine tower')
plt.xlabel('X-coordinate of impact location [m]')
plt.ylabel('Y-coordinate of impact location [m]')
circle = plt.Circle((0, 0), safety_distance, color='red', fill=False, linestyle='-', label='Recommended safety distance')
plt.gca().add_patch(circle)
plt.xlim(-350, 350)
plt.ylim(-350, 350)
plt.title('Impact locations of ice throws in xy-plane with counts > 1 per 10m x 10m')
plt.legend(loc='upper right')
plt.show()


# filter out the points with probability density > 1e-6
'''prob = df['Probability [m^-2]'].to_numpy()
df_f = df[(prob) > 1e-06]

x_impacts_f = df_f['X-coordinate [m]'].to_numpy()
y_impacts_f = df_f['Y-coordinate [m]'].to_numpy()

x_edges_f = np.arange(np.floor(np.min(x_impacts_f)), np.ceil(np.max(x_impacts_f)), bin_size)
y_edges_f = np.arange(np.floor(np.min(y_impacts_f)), np.ceil(np.max(y_impacts_f)), bin_size)

plt.figure(figsize=(10, 8))
_, _, _, hist_f = plt.hist2d(x_impacts_f, y_impacts_f, bins=[x_edges_f, y_edges_f], cmap='jet', cmin=1)
plt.colorbar(hist_f, label='Counts of ice strikes per 10m x 10m')
plt.scatter(0, 0, s=5, color='lightgrey', label='Turbine tower')
plt.xlabel('X-coordinate of impact location [m]')
plt.ylabel('Y-coordinate of impact location [m]')
circle_f = plt.Circle((0, 0), safety_distance, color='red', fill=False, linestyle='-', label='Recommended safety distance')
plt.gca().add_patch(circle_f)
plt.xlim(-350, 350)
plt.ylim(-350, 350)
plt.title('Impact locations of ice throws in xy-plane with counts per 10m x 10m with probabilities > 1e-6')
plt.legend(loc='upper right')
plt.show()'''