import matplotlib.pyplot as plt
import numpy as np

R = 45

fig, ax = plt.subplots(figsize=(6,6))
ax.set_aspect('equal')
ax.set_xlabel('x-axis [m]')
ax.set_ylabel('y-axis [m]')

ax.hlines(y=0, xmin=-2*R, xmax=2*R, color='black', lw=1)
ax.vlines(x=0, ymin=-2*R, ymax=2*R, color='black', lw=1)

'''x_axis = Axes.axhline(ax, y=0, xmin=-2*R, xmax=2*R, color='black', lw=1)
y_axis = Axes.axvline(ax, x=0, ymin=-2*R, ymax=2*R, color='black', lw=1)
ax.add_artist(x_axis)
ax.add_artist(y_axis)'''

ax.text(5, 80, 'N', fontsize=12, color='black')
ax.text(80, -10, 'E', fontsize=12, color='black')
ax.text(-10, -80, 'S', fontsize=12, color='black')
ax.text(-80, 5, 'W', fontsize=12, color='black')

#ax.annotate('', (0, 0), (-R, R), arrowprops={'arrowstyle':'-', 'shrinkA': 0, 'shrinkB': 0})
#ax.annotate('', (0, 0), (R, -R), arrowprops={'arrowstyle':'-', 'shrinkA': 0, 'shrinkB': 0})

'''ax.annotate('', (0, 0), (-R, R), arrowprops=dict(arrowstyle='-', shrinkA=0, shrinkB=0, color='grey', lw=2))
ax.annotate('', (0, 0), (R, -R), arrowprops=dict(arrowstyle='-', shrinkA=0, shrinkB=0, color='grey', l2=2))'''

ax.annotate('', (0, 0), ((R/5)+2, (-R/7)+2), arrowprops=dict(arrowstyle='-', shrinkA=0, shrinkB=0, color='grey'))
ax.annotate('', ((R/5)+2, (-R/7)+2), (R, -R), arrowprops=dict(arrowstyle='-', shrinkA=0, shrinkB=0, color='grey'))
ax.fill([0, (R/5)+2, R], [0, (-R/7)+2, -R], color='grey')
ax.annotate('', (0, 0), ((R/7)-2, (-R/5)-2), arrowprops=dict(arrowstyle='-', shrinkA=0, shrinkB=0, color='grey'))
ax.annotate('', ((R/7)-2, (-R/5)-2), (R, -R), arrowprops=dict(arrowstyle='-', shrinkA=0, shrinkB=0, color='grey'))
ax.fill([0, (R/7)-2, R], [0, (-R/5)-2, -R], color='grey')

ax.annotate('', (0, 0), ((-R/7)+2, (R/5)+2), arrowprops=dict(arrowstyle='-', shrinkA=0, shrinkB=0, color='grey'))
ax.annotate('', ((-R/7)+2, (R/5)+2), (-R, R), arrowprops=dict(arrowstyle='-', shrinkA=0, shrinkB=0, color='grey'))
ax.fill([0, (-R/7)+2, -R], [0, (R/5)+2, R], color='grey')
ax.annotate('', (0, 0), ((-R/5)-2, (R/7)-2), arrowprops=dict(arrowstyle='-', shrinkA=0, shrinkB=0, color='grey'))
ax.annotate('', ((-R/5)-2, (R/7)-2), (-R, R), arrowprops=dict(arrowstyle='-', shrinkA=0, shrinkB=0, color='grey'))
ax.fill([0, (-R/5)-2, -R], [0, (R/7)-2, R], color='grey')

ax.annotate('', (0, 90), (4, 86), arrowprops={'arrowstyle':'-', 'shrinkA': 0, 'shrinkB': 0})   # North right
ax.annotate('', (0, 90), (-4, 86), arrowprops={'arrowstyle':'-', 'shrinkA': 0, 'shrinkB': 0})  # North left
ax.annotate('', (90, 0), (86, 4), arrowprops={'arrowstyle':'-', 'shrinkA': 0, 'shrinkB': 0})   # East up
ax.annotate('', (90, 0), (86, -4), arrowprops={'arrowstyle':'-', 'shrinkA': 0, 'shrinkB': 0})  # East down
ax.annotate('', (0, -90), (4, -86), arrowprops={'arrowstyle':'-', 'shrinkA': 0, 'shrinkB': 0})  # South right
ax.annotate('', (0, -90), (-4, -86), arrowprops={'arrowstyle':'-', 'shrinkA': 0, 'shrinkB': 0}) # South left
ax.annotate('', (-90, 0), (-86, -4), arrowprops={'arrowstyle':'-', 'shrinkA': 0, 'shrinkB': 0}) # West down
ax.annotate('', (-90, 0), (-86, 4), arrowprops={'arrowstyle':'-', 'shrinkA': 0, 'shrinkB': 0})  # West up

ax.arrow(90, 90, -89, -89, head_width=4, head_length=4, fc='blue', ec='blue', length_includes_head=True, label='Wind direction')
#ax.annotate('', (90, 90), (0, 0), arrowprops=dict(arrowstyle='<-', shrinkA=0, shrinkB=0, color='blue', label='Wind direction'))

circ1 = ax.scatter(-R-5, R+5, facecolor='none', edgecolor='red', s=100)
in_plane = ax.scatter(-R-5, R+5, color='red', marker='x', s=10)
in_plane.set_label('Rotation in-plane')

circ2 = ax.scatter(R+5, -R-5, facecolor='none', edgecolor='red', s=100)
out_plane = ax.scatter(R+5, -R-5, color='red', s=10)
out_plane.set_label('Rotation out-of-plane')


theta = np.linspace(1, 0.5, 100)
x = np.cos(theta * np.pi/2) * 45
y = np.sin(theta * np.pi/2) * 45
ax.plot(x, y, lw=1, color='black')
ax.text(R/4, R +2, r'$\theta$ = $45^{\circ}$', fontsize=12, color='black')

ax.scatter(0,0, label='Tower', s=10, color='black', zorder=15)

ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
ax.set_xticks(np.arange(-90, 91, 45))
ax.set_yticks(np.arange(-90, 91, 45))

circ3_z = ax.scatter(-85, 85, facecolor='none', edgecolor='grey', s=100)
out_plane_z = ax.scatter(-85, 85, color='grey', s=10)
ax.text(-79, 82.6, r'z-axis', fontsize=12, color='grey')

ax.legend(loc='lower left')
ax.grid(which='both', linestyle='--', lw=0.2, color='grey')

plt.show()