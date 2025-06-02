
import matplotlib.pyplot as plt
import numpy as np

def draw_rotated_axes(ax, origin, rotation_matrix, length=1, labels=('x\'', 'y\'', 'z\''), color='green'):
    x, y, z = origin

    local_axes = np.array([[length, 0, 0], [0, length, 0], [0, 0, length]])
    rotated_axes = np.dot(rotation_matrix, local_axes.T).T

    for i, (dx, dy, dz) in enumerate(rotated_axes):
        ax.quiver(x, y, z, dx, dy, dz, color=color, arrow_length_ratio=0.1)
        ax.text(x + dx, y + dy, z + dz, labels[i], color=color, fontsize=12)

def draw_axes(ax, origin, length=1.5, labels=('X', 'Y', 'Z'), color='k'):
    x, y, z = origin
    ax.quiver(x, y, z, length, 0, 0, color=color, arrow_length_ratio=0.1)
    ax.quiver(x, y, z, 0, length, 0, color=color, arrow_length_ratio=0.1)
    ax.quiver(x, y, z, 0, 0, length, color=color, arrow_length_ratio=0.1)
    ax.text(x + length, y, z, labels[0], color=color, fontsize=12)
    ax.text(x, y + length, z, labels[1], color=color, fontsize=12)
    ax.text(x, y, z + length, labels[2], color=color, fontsize=12)

def rotation_matrix_z(angle_degrees):
    angle_radians = np.radians(angle_degrees)
    return np.array([[np.cos(angle_radians), -np.sin(angle_radians), 0],
                     [np.sin(angle_radians),  np.cos(angle_radians), 0],
                     [0, 0, 1]])

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

z = 3      # Height at which wind speed is depicted
U_z = 1.5  # Length of U(z) vector
ax.quiver(0, 0, z, U_z, 0, 0, color='blue', alpha=0.7, arrow_length_ratio=0.1)
ax.text(U_z + 0.1, 0, z, 'U(z)', color='blue', fontsize=12)

draw_axes(ax, (0, 0, 0), length=1.5, labels=('X', 'Y', 'Z'))

draw_axes(ax, (1, 1, 2), length=1, labels=('x', 'y', 'z'), color='green')

rotation = rotation_matrix_z(45)  # 45-degree rotation around the z-axis
draw_rotated_axes(ax, (1, 1, 2), rotation, length=1, labels=('u', 'v', 'w'), color='orange')

ice_fragment_x = [0.9, 1.1, 1.2, 1.0, 0.8]
ice_fragment_y = [1.0, 1.2, 1.1, 0.9, 1.1]
ice_fragment_z = [2.1, 2.2, 2.0, 1.9, 2.0]
ax.plot_trisurf(ice_fragment_x, ice_fragment_y, ice_fragment_z, color='gray', alpha=0.7)

ax.set_xlim([0, 3])
ax.set_ylim([0, 3])
ax.set_zlim([0, 5])
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
#ax.set_title("Ice Fragment with Rotated Local Coordinate System")

plt.show()