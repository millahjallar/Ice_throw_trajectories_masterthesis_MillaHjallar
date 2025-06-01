import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.stats as st
plt.style.use('seaborn-v0_8-whitegrid')

df = pd.read_csv('Ice_thrower_simulation_100000its_20.05.2025.csv')

x_impacts = df['X-coordinate [m]'].to_numpy()
y_impacts = df['Y-coordinate [m]'].to_numpy()

positions = np.vstack((x_impacts, y_impacts))

kde = st.gaussian_kde(positions)

#x_min, x_max = -400, 400
#y_min, y_max = -400, 400

x_min, x_max = -240, 240
y_min, y_max = -240, 240

#x_grid = np.linspace(x_min, x_max, 801)  # 10 x 10 grid spacing
#y_grid = np.linspace(y_min, y_max, 801)

x_grid = np.linspace(x_min, x_max, 481) # 1x1m grid spacing
y_grid = np.linspace(y_min, y_max, 481)
X, Y = np.meshgrid(x_grid, y_grid)

coords_for_kde = np.vstack([X.ravel(), Y.ravel()])

dx = x_grid[1] - x_grid[0]
dy = y_grid[1] - y_grid[0]
cell_area = dx * dy

#Z = kde(coords_for_kde).reshape(X.shape)
Z = cell_area * kde(coords_for_kde).reshape(X.shape)
Z_masked = np.ma.masked_where(Z < 1e-6, Z)

fig, ax = plt.subplots(figsize=(8, 6))

levels = np.logspace(-6, -4, 5)

#cmap = plt.get_cmap('Blues', len(levels)-1)
cmap = plt.get_cmap('Blues')

fig, ax = plt.subplots(figsize=(8, 6))
cont = ax.contourf(
    X, Y, Z_masked,
    levels=levels,
    cmap=cmap,
    norm=colors.LogNorm(vmin=levels[0], vmax=levels[-1])
)

cbar = fig.colorbar(
    cont, ax=ax,
    boundaries=levels,
    spacing='uniform',
    ticks=levels,
    format='%.0e'
)
cbar.set_label('Probability density [$m^{-2}$]')

ax.scatter(0, 0, s=20, c='red', label='Turbine tower', zorder=5)
ax.set(xlabel='X-coordinate [m]', ylabel='Y-coordinate [m]',
       xlim=(x_min, x_max), ylim=(y_min, y_max),
       title='Ice-strike probability density per $m^2$')

ax.set_xlabel('X-coordinate [m]')
ax.set_ylabel('Y-coordinate [m]')
ax.set_title('Ice-strike probability density per $m^2$')
plt.show()