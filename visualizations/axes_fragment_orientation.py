import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Arc
plt.style.use('seaborn-v0_8-whitegrid')

state  = [0, 0, 0,  15, 0, -5]   # vx=15 m/s downwind, vz=−5 m/s up
Cd, Cl = 1.0, 0.5
A_ref  = 0.1                     # m²
wspd   = 10                      # m/s
wdir   = 0                       # coming from north
RHO    = 1.225  # kg/m³

alph = np.deg2rad(270 - wdir)
u = wspd * np.cos(alph)
Vx_rel = state[3] - u
Vz_rel = state[5]
V_rel = np.array([Vx_rel, Vz_rel])
dir_rel  = V_rel / np.linalg.norm(V_rel)
dir_drag = -dir_rel
dir_lift = np.array([-dir_rel[1], dir_rel[0]])

L   = 0.7   # total length of fragment
h   = 0.15  # thickness

corners = []
for s1 in [+1, -1]:
    for s2 in [+1, -1]:
        corner = (s1 * (L/2) * dir_rel) + (s2 * (h/2) * dir_lift)
        corners.append(tuple(corner))

corners = [corners[0], corners[1], corners[3], corners[2]]

fig, ax = plt.subplots(figsize=(5,5))
ax.set_aspect('equal')
ax.axis('off')

R = 1.0

# axes
ax.annotate('', xy=(1.2, -0.8), xytext=(-1.2, -0.8), arrowprops=dict(arrowstyle='->', lw=1, color='black', shrinkA=0, shrinkB=0))
ax.annotate('', xy=(-1.2,  1.2), xytext=(-1.2, -0.8), arrowprops=dict(arrowstyle='->', lw=1, color='black', shrinkA=0, shrinkB=0))

ax.text(0.0,  -0.85, 'horizontal axis', ha='center', va='top', fontsize=10)
mid_y = (-0.8 + 1.2) / 2
ax.text(-1.25, mid_y, 'vertical axis',
        rotation=90, ha='center', va='center', fontsize=10)

ax.arrow(0, 0, -R, 0, head_width=0.05, length_includes_head=True, color='black')
#ax.text(-R, 0, r'$u_h - U$', va='center', ha='right')
ax.text(-R, 0, r'$V_H$', va='center', ha='right')
ax.arrow(0, 0, 0, R, head_width=0.05, length_includes_head=True, color='black')
ax.text(0, R, r'$w$', va='bottom', ha='center')

ax.arrow(0,0, *dir_rel, head_width=0.05, length_includes_head=True, color='red')
ax.text(*(dir_rel)*1.1, r'$\vec F_D$', color='red')

ax.arrow(0,0, *dir_drag, head_width=0.05, length_includes_head=True, color='gray')
ax.text(*(dir_drag)*1.1, r'$\vec V_{\rm rel}$', color='gray')

ax.arrow(0,0, *(dir_lift*0.5), head_width=0.05, length_includes_head=True, color='blue')
ax.text(*(dir_lift*0.55), r'$\vec F_L$', color='blue')

# ice fragment
rect = Polygon(corners, closed=True, edgecolor='gray', facecolor='lightgray', lw=1, alpha=0.8)
ax.add_patch(rect)
ax.text(0, 0, 'ice', ha='center', va='center')

# Angle between V_H and V_rel
angle_horiz = 180.0
angle_rel   = np.degrees(np.arctan2(dir_drag[1], dir_drag[0]))

r_arc = 0.5

arc = Arc((0, 0), 2*r_arc, 2*r_arc, angle=0, theta1=angle_rel, theta2=angle_horiz, lw=1, color='black')
ax.add_patch(arc)

mid_angle = (angle_rel + angle_horiz) / 2
mid_rad   = np.deg2rad(mid_angle)
label_r   = r_arc + 0.05
ax.text(np.cos(mid_rad)*label_r, np.sin(mid_rad)*label_r, r'$\mathcal{X}$', ha='center', va='center', fontsize=12)

ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-0.8, 1.2)

plt.tight_layout()
plt.show()
