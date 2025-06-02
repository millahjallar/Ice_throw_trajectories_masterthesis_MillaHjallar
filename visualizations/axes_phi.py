import matplotlib.pyplot as plt
import numpy as np

R = 45
arrow_length = 45

fig, ax = plt.subplots(figsize=(6,6))
ax.set_aspect('equal')
ax.set_xlabel("x-axis (m)")
ax.set_ylabel("z-axis (m)")
#ax.set_title("Circle with 4 Tangential Arrows\n(Counterclockwise Rotation)")

circle = plt.Circle((0, 0), R, fill=False, linestyle='--', color='gray')
ax.add_artist(circle)

positions = {
    'φ=0': (-R, 0),
    'φ=90': (0, R),
    'φ=180': (R, 0),
    'φ=270': (0, -R)
}

tangent_vectors = {}
for label, (x, y) in positions.items():
    theta = np.arctan2(y, x)
    tx = np.sin(theta)
    ty = -np.cos(theta)
    tangent_vectors[label] = (tx * arrow_length, ty * arrow_length)

for label, (x, y) in positions.items():
    dx, dy = tangent_vectors[label]
    ax.arrow(x, y, dx, dy,
             head_width=2, head_length=2,
             fc='blue', ec='blue', length_includes_head=True)
    
    ax.text(x * 1.2, y * 1.2, label, fontsize=12, ha='center', va='center')
    ax.arrow(0, 0, R, 0, head_width=2, head_length=2, fc='red', ec='red', length_includes_head=True)
    ax.text(R/2, 2, 'R', fontsize=12, ha='center', color='red', va='center')

ax.set_xlim(-R - 20, R + 20)
ax.set_ylim(-R - 20, R + 20)
circ3_z = ax.scatter(50, 50, facecolor='none', edgecolor='black', s=100)
out_plane_z = ax.scatter(50, 50, color='black', s=10, marker='x')
ax.text(31.7, 48.6, r'y-axis', fontsize=12, color='black')

plt.show()