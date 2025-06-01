import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.stats as st
import matplotlib.ticker as mtick
import numpy as np
np.random.seed(12345)

df = pd.read_csv('Ice_thrower_simulation_100000its_20.05.2025.csv')

x_impacts = df['X-coordinate [m]'].to_numpy()
y_impacts = df['Y-coordinate [m]'].to_numpy()

turbine_sweref = [
    (663569.7933, 7129066.6957),  # T1 (reference)
    (663915.9511, 7128869.9257),  # T2
    (664456.4301, 7128719.0512),  # T3
    (664730.3689, 7128455.6593),  # T4
    (664010.5406, 7128452.4760),  # T5
    (664251.9642, 7128106.5805),  # T6
    (664624.6554, 7127970.0834),  # T7
    (665077.5062, 7128029.8140),  # T8
    (665217.6414, 7127747.2348),  # T9
    (664761.8543, 7127448.3634),  # T10
    (662832.1426, 7128594.4370),  # T11
    (663078.8921, 7128358.9620),  # T12
    (663336.5629, 7128091.5835),  # T13
    (663551.8850, 7127824.4047),  # T14
    (663854.3927, 7127544.8761),  # T15
    (662429.5312, 7128204.0063),  # T16
    (662752.9751, 7127916.1524),  # T17
    (662549.6188, 7127522.5421),  # T18
    (662449.6879, 7126962.7519),  # T19
    (663017.6234, 7126858.8081),  # T20
    (663334.4010, 7126643.3477),  # T21
    (662610.7725, 7126606.7900),  # T22
    (662803.1000, 7126276.1112),  # T23
    (663138.9706, 7126077.6548),  # T24
    (663434.9501, 7125826.1436),  # T25
    (663598.6459, 7125454.7752),  # T26
    (663780.6513, 7125097.8609),  # T27
    (663958.4941, 7124692.8840),  # T28
    (663356.9500, 7124889.4356),  # T29
    (662939.9459, 7125073.8526),  # T30
    (663000.6962, 7125538.2744),  # T31
    (664444.1746, 7124794.3240),  # T32
    (664807.7246, 7125111.8696),  # T33
    (664790.1062, 7125584.9239),  # T34
    (664227.5776, 7125147.6747),  # T35
    (664182.6484, 7125545.6166),  # T36
    (663937.7575, 7125872.6857),  # T37
    (663860.5101, 7126305.8951),  # T38
    (663775.0755, 7126695.2659),  # T39
    (664240.7722, 7126836.9656),  # T40
    (668693.2441, 7121867.6109),  # T41
    (668265.8110, 7121971.9911),  # T42
    (668506.7982, 7122452.0138),  # T43
    (667878.0430, 7122314.7766),  # T44
    (668152.5788, 7122630.1455),  # T45
    (667641.8045, 7122868.0284),  # T46
    (668015.2187, 7123181.7221),  # T47
    (668484.7911, 7123008.2777),  # T48
    (668620.4381, 7123378.7959),  # T49
    (668286.0916, 7123623.8397),  # T50
    (667880.6052, 7123700.9177),  # T51
    (667937.6868, 7124363.2855),  # T52
    (667488.8831, 7124589.9178),  # T53
    (667675.5971, 7125005.6069),  # T54
    (667954.5611, 7125298.3039),  # T55
    (668372.9134, 7125100.9240),  # T56
    (668863.8841, 7125061.4019),  # T57
    (668984.5217, 7124584.0304),  # T58
    (668791.7451, 7124229.2721),  # T59
    (668325.7672, 7126014.9775),  # T60
    (667955.8390, 7126203.1465),  # T61
    (668260.8850, 7126697.7460),  # T62
    (668578.7022, 7127013.1910),  # T63
    (669041.8531, 7126650.8442),  # T64
    (669498.9719, 7126338.9720),  # T65
    (669232.8010, 7126001.9352),  # T66
    (669139.0263, 7127086.6228),  # T67
    (668864.4118, 7127369.6493),  # T68
    (668938.2123, 7127765.3072),  # T69
    (669573.3899, 7127886.8045),  # T70
    (669824.1364, 7127514.9951),  # T71
    (670320.9417, 7127633.2136),  # T72
    (670671.9397, 7127324.5407),  # T73
    (670269.4626, 7126986.4427),  # T74
]

#minlon, maxlon, minlat, maxlat = (18.3118, 18.5451, 64.1768, 64.2544)
minlon, maxlon, minlat, maxlat = (660456.5331, 672232.8406, 7121512.4717, 7129543.5361)

#minlon, maxlon, minlat, maxlat = (660456.5331, 672232.8406,7121512.4717, 7129543.5361)
png_SWEREF = [
    (660456.5331, 7121512.4717),  # min lon and lat
    (672232.8406, 7129543.5361)   # max lon and lat
]

ref_turbine = turbine_sweref[0]
ref_x, ref_y = ref_turbine

turbine_sweref_centered = [(x - ref_x, y - ref_y) for (x, y) in turbine_sweref]
png_coordinates_centered = [(x - ref_x, y - ref_y) for (x, y) in png_SWEREF]
(min_x, min_y) = png_coordinates_centered[0]
(max_x, max_y) = png_coordinates_centered[1]

all_turbine_impacts = {}
for i, (turbine_x, turbine_y) in enumerate(turbine_sweref_centered):
    # Shifting the reference impact coordinates by the turbine's position
    turbine_x_impacts = x_impacts + turbine_x
    turbine_y_impacts = y_impacts + turbine_y
    all_turbine_impacts[i] = (turbine_x_impacts, turbine_y_impacts)

fig, ax = plt.subplots(figsize=(8, 6))

arr_img = plt.imread('new_map.png')
ax.imshow(
    arr_img,
    extent=(min_x, max_x, min_y, max_y),
    origin='upper',
    zorder=0,
    alpha=0.9,
)

ref_positions = np.array(all_turbine_impacts[0])
kde = st.gaussian_kde(ref_positions, bw_method='scott')

#x_min, x_max = -400, 400
#y_min, y_max = -400, 400

x_min, x_max = -240, 240
y_min, y_max = -240, 240

#x_grid = np.linspace(x_min, x_max, 801)  # 10 x 10 grid spacing
#y_grid = np.linspace(y_min, y_max, 801)

x_grid = np.linspace(x_min, x_max, 481) # 1x1m grid spacing
y_grid = np.linspace(y_min, y_max, 481)
X, Y = np.meshgrid(x_grid, y_grid)

dx = x_grid[1] - x_grid[0]
dy = y_grid[1] - y_grid[0]
cell_area = dx * dy

coords_for_kde = np.vstack([X.ravel(), Y.ravel()])
Z = cell_area * kde(coords_for_kde).reshape(X.shape)
Z_masked = np.ma.masked_where(Z < 1e-6, Z)

num_turbines = len(all_turbine_impacts)
for i in range(num_turbines):
    x_i, y_i = turbine_sweref[i]
    dx = x_i - ref_x
    dy = y_i - ref_y
    X_shifted = X + dx
    Y_shifted = Y + dy

    cont = ax.contourf(
        X_shifted, Y_shifted, Z_masked,
        levels=50,
        cmap='Blues',
        norm=colors.LogNorm(),
        zorder=4,
        alpha=0.8,
    )

'''ax.set_title('Probability of ice strikes per m$^{2}$')
ax.tick_params(labelbottom=False, labelleft=False)

# Define conversion functions between local (meter) coordinates and degrees.
def x_to_lon(x):
    return (x - min_x) / (max_x - min_x) * (maxlon - minlon) + minlon

def lon_to_x(lon):
    return (lon - minlon) / (maxlon - minlon) * (max_x - min_x) + min_x

def y_to_lat(y):
    return (y - min_y) / (max_y - min_y) * (maxlat - minlat) + minlat

def lat_to_y(lat):
    return (lat - minlat) / (maxlat - minlat) * (max_y - min_y) + min_y

# Create secondary axes that use these conversion functions.
secax_x = ax.secondary_xaxis('bottom', functions=(x_to_lon, lon_to_x))
secax_y = ax.secondary_yaxis('left', functions=(y_to_lat, lat_to_y))

secax_x.set_xlabel('Longitude [°]')
secax_y.set_ylabel('Latitude [°]')

# Set tick positions for the secondary axes (in degrees)
secax_x.set_xticks(np.linspace(minlon, maxlon, 8))
secax_y.set_yticks(np.linspace(minlat, maxlat, 8))

secax_x.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
secax_y.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

cbar = fig.colorbar(cont, ax=ax)
cbar.set_label('Probability Density [m$^{-2}$]')
ax.grid(True)
plt.show()'''

#ax.set_title('Ice-strike probability density per $10 m^2$ x $10 m^2$ overlaid on map')
ax.set_title('Ice-strike probability density per $m^2$ overlaid on map')
ax.set_xlabel('X-coordinate of impact location [m]')
ax.set_ylabel('Y-coordinate of impact location [m]')

ax.set_xticks(np.linspace(min_x, max_x, 8))
ax.set_yticks(np.linspace(min_y, max_y, 8))

for i in range(1, num_turbines):
    x_i, y_i = turbine_sweref_centered[i]
    ax.scatter(x_i, y_i, c='red', s=0.1, zorder=5)

ax.scatter(0, 0, s=0.1, c='red', zorder=5, label='Turbine tower')
ax.legend(loc='upper right', fontsize=8)
# plotting the reference turbine tower

cbar = fig.colorbar(cont, ax=ax, shrink=0.75)
#cbar.set_label('Probability density per $10 m^2$ x $10 m^2$')
cbar.set_label('Probability density per $m^2$')
ax.grid(True)
plt.savefig('map_contourfs.pdf', format='pdf')
plt.show()
