'''Rolv: Here, we consider ice pieces with impact kinetic energies
above 40 J and with weights above 100 g, as dangerous (fatal)
[2][4][18]'''

import numpy as np
import matplotlib.pyplot as plt
from A_scatter_KDE_1m2 import mass_distribution, areas
import pandas as pd
import matplotlib.colors as mcolors

RATED_WIND_SPEED = 12.0           # Rated wind speed [m/s]
CUT_IN_WIND_SPEED = 3.0           # Cut-in wind speed [m/s]
CUT_OUT_WIND_SPEED = 28.0         # Cut-out wind speed [m/s]
R = 45.0                          # Rotor radius [m]

df = pd.read_csv('Ice_thrower_simulation_100000its_12.05.2025.csv')
wspd = df['wind speed [m/s]'].to_numpy()
x_impacts = df['X-coordinate [m]'].to_numpy()
y_impacts = df['Y-coordinate [m]'].to_numpy()
impact_v = df['Impact velocity [m/s]'].to_numpy()

def kinetic_energy(mass, velocity):
    return 0.5 * mass * velocity**2

energy = kinetic_energy(mass_distribution, impact_v)

mask = (energy > 40) & (mass_distribution > 0.1)

energy_below_40 = energy[~mask]
x_impacts_below_40 = x_impacts[~mask]
y_impacts_below_40 = y_impacts[~mask]

x_masked = x_impacts[mask]
y_masked = y_impacts[mask]
energy_masked = energy[mask]

max_E = energy_masked.max()
edges = np.concatenate(([40], np.arange(100, np.ceil(max_E/100)*100 + 100, 300)))

cmap = plt.get_cmap('viridis', len(edges) - 1)
norm = mcolors.BoundaryNorm(edges, ncolors=cmap.N, clip=False)

max_E_all_throws = energy.max()
edges_all_throws = np.concatenate(([0], np.arange(100, np.ceil(max_E_all_throws/100)*100 + 100, 300)))

cmap_all_throws = plt.get_cmap('viridis', len(edges_all_throws) - 1)
norm_all_throws = mcolors.BoundaryNorm(edges_all_throws, ncolors=cmap_all_throws.N, clip=False)

print(len(x_impacts))
print(len(x_masked))

fig, ax = plt.subplots(1, 2)
'''scatter = ax[0].scatter(x_impacts, y_impacts, s=1, c=energy, cmap=cmap_all_throws,
                      norm=norm_all_throws, label='All throws', alpha=0.3)
cbar = plt.colorbar(scatter, boundaries=edges_all_throws, ticks=edges_all_throws)
cbar.set_label('Energy [J]')
ax[0].scatter(0, 0, s=20, c='red', label='Initial position of ice throw', zorder=5)
ax[0].set_xlim(-270, 270)
ax[0].set_ylim(-270, 270)
ax[0].set_xlabel('X-coordinate [m]')
ax[0].set_ylabel('Y-coordinate [m]')
ax[0].set_title('All ice throws in the xy-plane')
ax[0].legend(loc='best')
ax[0].grid()'''

scatter = ax[0].scatter(x_impacts_below_40, y_impacts_below_40, s=1, c=energy_below_40, label='All throws', alpha=0.3)
cbar = plt.colorbar(scatter)
cbar.set_label('Energy [J]')
ax[0].scatter(0, 0, s=20, c='red', label='Initial position of ice throw', zorder=5)
ax[0].set_xlim(-270, 270)
ax[0].set_ylim(-270, 270)
ax[0].set_xlabel('X-coordinate [m]')
ax[0].set_ylabel('Y-coordinate [m]')
ax[0].set_title('Only ice throws with $E_K$<40J in the xy-plane')
ax[0].legend(loc='best')
ax[0].grid()

scatter1 = ax[1].scatter(x_masked, y_masked, s=1, c=energy_masked, cmap=cmap,
                      norm=norm, label='Fatal throws')
cbar = plt.colorbar(scatter1, boundaries=edges, ticks=edges)
cbar.set_label('Energy [J]')
ax[1].scatter(0, 0, s=20, c='red', label='Initial position of ice throw', zorder=5)
ax[1].set_xlim(-270, 270)
ax[1].set_ylim(-270, 270)
ax[1].set_xlabel('X-coordinate [m]')
ax[1].set_ylabel('Y-coordinate [m]')
ax[1].set_title('Only ice throws with kinetic energy > 40J (fatal) in the xy-plane')
ax[1].legend(loc='best')
ax[1].grid()
plt.show()


fig, ax = plt.subplots(1, 2)
ax[0].scatter(x_impacts, y_impacts, s=1, label='All throws', alpha=0.3) #row=0, col=0
ax[0].scatter(0, 0, s=20, c='red', label='Initial position of ice throw', zorder=5)
ax[0].set_xlim(-270, 270)
ax[0].set_ylim(-270, 270)
ax[0].set_xlabel('X-coordinate [m]')
ax[0].set_ylabel('Y-coordinate [m]')
ax[0].set_title('All ice throws in the xy-plane')
ax[0].legend(loc='best')
ax[0].grid()

ax[1].scatter(x_masked, y_masked, s=1, label='Fatal throws', alpha=0.3) #row=1, col=0
ax[1].scatter(0, 0, s=20, c='red', label='Initial position of ice throw', zorder=5)
ax[1].set_xlim(-270, 270)
ax[1].set_ylim(-270, 270)
ax[1].set_xlabel('X-coordinate [m]')
ax[1].set_ylabel('Y-coordinate [m]')
ax[1].set_title('Only ice throws with kinetic energy > 40J (fatal) in the xy-plane')
ax[1].legend(loc='best')
ax[1].grid()
plt.show()

'''plt.figure(figsize=(8, 6))
scatter = plt.scatter(x_masked, y_masked, s=1, c=energy_masked, cmap=cmap,
                      norm=norm, label='Fatal throws')
cbar = plt.colorbar(scatter, boundaries=edges, ticks=edges)
cbar.set_label('Energy [J]')
plt.scatter(0, 0, s=20, c='red', label='Initial position of ice throw', zorder=5)
plt.xlim(-250, 250)
plt.ylim(-250, 250)
plt.xlabel('X-coordinate [m]')
plt.ylabel('Y-coordinate [m]')
plt.title('Ice throws with fatal kinetic energy (> 40 J) in the xy-plane')
plt.legend(loc='best')
plt.grid()
plt.show()'''
