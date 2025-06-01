import random
import numpy as np
np.random.seed(12345)
import pandas as pd
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
from A_scatter_KDE_1m2 import WSPD, WDIR, area_distribution, mass_distribution

N = 10000                        # number of simulations
new_csv = False

# -------------------------------------------------------
# ---------------------- Constants ----------------------
# -------------------------------------------------------

HUB_H = 95.0                      # Hub height [m]
R = 45.0                          # Ejection radius [m]
RHO_AIR = 1.225                   # Density of air [kg/m^3]
RHO_ICE = 520.0                   # Density of ice (Snow ice blend) [kg/m^3]
C_D = 1.7                         # Drag coefficient (flat-shaped ice)
C_L = 1.0                         # Lift coefficient (flat-shaped ice)
RATED_WIND_SPEED = 12.0           # Rated wind speed [m/s]
CUT_IN_WIND_SPEED = 3.0           # Cut-in wind speed [m/s]
CUT_OUT_WIND_SPEED = 28.0         # Cut-out wind speed [m/s]
g = 9.81                          # Gravity [m/s^2]

VON_K = 0.41                      # von Karman constant
z0 = 0.03                         # Roughness length (assumed)

# ---------------------- Defining functions ----------------------

'''def rotor_speed(wind_speed):
    return MAX_OMEGA_ROTOR_SPEED * np.clip(wind_speed / MAX_WIND_SPEED, 0, 1)'''

def wind_profile(z, u_star):
    z_eff = max(z, z0)
    return (u_star / VON_K) * np.log(z_eff / z0)

# Friction velocity
def U_star(wspd):
    return (VON_K * wspd) / np.log(HUB_H / z0)

def rotor_speed(wind_speed):
    MAX_OMEGA_ROTOR_SPEED = (7 * RATED_WIND_SPEED) / R 
    
    if wind_speed < CUT_IN_WIND_SPEED:
        return 0.0
    elif wind_speed < RATED_WIND_SPEED:
        fraction = (wind_speed - CUT_IN_WIND_SPEED) / (RATED_WIND_SPEED - CUT_IN_WIND_SPEED)
        return fraction * MAX_OMEGA_ROTOR_SPEED
    elif wind_speed <= CUT_OUT_WIND_SPEED:
        return MAX_OMEGA_ROTOR_SPEED
    else:
        return 0.0


def wind_components(wspd, wdir):
    theta = np.radians(wdir)
    '''
    theta = 0deg:  wspd = (0, -wspd)
    theta = 90deg: wspd = (-wspd, 0)
    '''
    wx = -wspd * np.sin(theta)
    wy = -wspd * np.cos(theta) 
    return wx, wy

def eq_of_biswas(t, state, mass, Cd, Cl, A_ref, wspd, wdir, u_star):
    x, y, z, vx, vy, vz = state

    wspd = wind_profile(z, u_star)
    u, v = wind_components(wspd, wdir)

    V_rel_x = vx - u
    V_rel_y = vy - v
    V_rel_z = vz
    V_rel = np.sqrt(V_rel_x**2 + V_rel_y**2 + V_rel_z**2)
    V_rel_H = np.sqrt(V_rel_x**2 + V_rel_y**2)

    F_D = 0.5 * RHO_AIR * Cd * A_ref * (V_rel**2)
    F_L = 0.5 * RHO_AIR * Cl * A_ref * (V_rel**2)

    xsi_sin = V_rel_z / V_rel
    xsi_cos = V_rel_H / V_rel

    drag_x = -F_D * (V_rel_x / V_rel_H) * xsi_cos
    drag_y = -F_D * (V_rel_y / V_rel_H) * xsi_cos
    drag_z = -F_D * xsi_sin

    lift_x = F_L * (V_rel_x / V_rel_H) * xsi_sin
    lift_y = F_L * (V_rel_y / V_rel_H) * xsi_sin
    lift_z = F_L * xsi_cos

    ax = (lift_x + drag_x) / mass
    ay = (lift_y + drag_y) / mass
    az = -g + (lift_z + drag_z) / mass

    return np.array([vx, vy, vz, ax, ay, az])

def RK4(state0, mass, Cd, Cl, area, wspd, wdir, u_star, dt=0.01, t_max=100.0):
    t = 0.0
    state = np.array(state0)

    while t < t_max:
        if state[2] <= 0:
            break
        k1 = eq_of_biswas(t, state, mass, Cd, Cl, area, wspd, wdir, u_star)
        k2 = eq_of_biswas(t + dt/2.0, state + dt*k1/2.0, mass, Cd, Cl, area, wspd, wdir, u_star)
        k3 = eq_of_biswas(t + dt/2.0, state + dt*k2/2.0, mass, Cd, Cl, area, wspd, wdir, u_star)
        k4 = eq_of_biswas(t + dt, state + dt*k3, mass, Cd, Cl, area, wspd, wdir, u_star)
        state = state + ((dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4))
        t += dt

    return state

def trajectory(omega, mass, blade_angle_deg, area, wspd, wdir, u_star):
    phi = np.deg2rad(blade_angle_deg)
    V_t = omega * R

    x_00 = -R * np.cos(phi)
    y_00 = 0.0
    z_00 = HUB_H + R * np.sin(phi)

    vx_00 = V_t * np.sin(phi)
    vy_00 = 0.0
    vz_00 = V_t * np.cos(phi)

    yaw = np.deg2rad(wdir)

    x0 = x_00 * np.cos(yaw) + y_00 * np.sin(yaw)
    y0 = -x_00 * np.sin(yaw) + y_00 * np.cos(yaw)
    z0 = z_00

    vx0 = vx_00 * np.cos(yaw) + vy_00 * np.sin(yaw)
    vy0 = -vx_00 * np.sin(yaw) + vy_00 * np.cos(yaw)
    vz0 = vz_00

    state0 = [x0, y0, z0, vx0, vy0, vz0]
    final_state = RK4(state0, mass, C_D, C_L, area, wspd, wdir, u_star)

    return final_state[0], final_state[1]


x_impacts = np.zeros(N)
y_impacts = np.zeros(N)

def main():
    #OMEGA_ROTOR_SPEED = rotor_speed(WSPD)
    random_angles = np.random.uniform(0, 360, N)
    ejection_angles = np.zeros(N)
    u_stars = np.zeros(N)

    for i in tqdm(range(N)):
        angle = random_angles[i]
        ejection_angles[i] = angle
        wspd_i = WSPD[i]
        wdir_i = WDIR[i]
        u_star_i = U_star(wspd_i)
        u_stars[i] = u_star_i
        omega_i = rotor_speed(wspd_i)

        x_impacts[i], y_impacts[i] = trajectory(
            omega_i,
            mass_distribution[i],
            angle,
            area_distribution[i],
            wspd_i,
            wdir_i,
            u_star_i
        )

    process_results(ejection_angles)

file_path = 'IceThrower_dataBase_2013_2016.xlsx'

info_data_new = pd.read_excel(file_path, sheet_name='Info', skiprows=4)
dv_data_new = pd.read_excel(file_path, sheet_name='DV', skiprows=4, decimal=',')
vf1_data_new = pd.read_excel(file_path, sheet_name='VF 1', skiprows=4, decimal=',')
vf2_data_new = pd.read_excel(file_path, sheet_name='VF 2', skiprows=4, decimal=',')
vf3_data_new = pd.read_excel(file_path, sheet_name='VF 3', skiprows=4, decimal=',')
vf4_data_new = pd.read_excel(file_path, sheet_name='VF 4', skiprows=4, decimal=',')
vf5_data_new = pd.read_excel(file_path, sheet_name='VF 5', skiprows=4, decimal=',')
vf6_data_new = pd.read_excel(file_path, sheet_name='VF 6', skiprows=4, decimal=',')
SK_data_new = pd.read_excel(file_path, sheet_name='SK', skiprows=4, decimal=',')

# --------- Cleaning data ---------
dv_data_new = dv_data_new[(dv_data_new['X-coordinate'] != 'N') & (dv_data_new['Y-coordinate'] != 'N')]
vf1_data_new = vf1_data_new[(vf1_data_new['X-coordinat.1'] != 'N') & (vf1_data_new['Y-coordinat.1'] != 'N')]
vf2_data_new = vf2_data_new[(vf2_data_new['X-coordinate.1'] != 'N') & (vf2_data_new['Y-coordinate.1'] != 'N')]
vf3_data_new = vf3_data_new[(vf3_data_new['X-coordinate.1'] != 'N') & (vf3_data_new['Y-coordinate.1'] != 'N')]
vf4_data_new = vf4_data_new[(vf4_data_new['X-coordinate.1'] != 'N') & (vf4_data_new['Y-coordinate.1'] != 'N')]
vf5_data_new = vf5_data_new[(vf5_data_new['X-coordinate'] != 'N') & (vf5_data_new['Y-coordinate'] != 'N')]
vf6_data_new = vf6_data_new[(vf6_data_new['X-coordinate.1'] != 'N') & (vf6_data_new['Y-coordinate.1'] != 'N')]

dv_data_new = dv_data_new[(dv_data_new['X-coordinate'].notna()) & (dv_data_new['Y-coordinate'].notna())]
vf1_data_new = vf1_data_new[(vf1_data_new['X-coordinat.1'].notna()) & (vf1_data_new['Y-coordinat.1'].notna())]
vf2_data_new = vf2_data_new[(vf2_data_new['X-coordinate.1'].notna()) & (vf2_data_new['Y-coordinate.1'].notna())]
vf3_data_new = vf3_data_new[(vf3_data_new['X-coordinate.1'].notna()) & (vf3_data_new['Y-coordinate.1'].notna())]
vf4_data_new = vf4_data_new[(vf4_data_new['X-coordinate.1'].notna()) & (vf4_data_new['Y-coordinate.1'].notna())]
vf5_data_new = vf5_data_new[(vf5_data_new['X-coordinate'].notna()) & (vf5_data_new['Y-coordinate'].notna())]
vf6_data_new = vf6_data_new[(vf6_data_new['X-coordinate.1'].notna()) & (vf6_data_new['Y-coordinate.1'].notna())]
SK_data_new = SK_data_new[(SK_data_new['(m)'] != 'N') & (SK_data_new['X-coordinate'] != 'N') & (SK_data_new['Y-coordinate'] != 'N')]

# --------- Extracting data for each turbine ---------
dv_distance_X_T1 = dv_data_new.loc[dv_data_new['WTG ID'] == 1, 'X-coordinate']
vf1_distance_X_T5 = vf1_data_new.loc[vf1_data_new['WTG ID'] == 5, 'X-coordinat.1']
vf1_distance_X_T13 = vf1_data_new.loc[vf1_data_new['WTG ID'] == 13, 'X-coordinat.1']
vf2_distance_X_T6 = vf2_data_new.loc[vf2_data_new['WTG ID'] == 6, 'X-coordinate.1']
vf2_distance_X_T11 = vf2_data_new.loc[vf2_data_new['WTG ID'] == 11, 'X-coordinate.1']
vf2_distance_X_T12 = vf2_data_new.loc[vf2_data_new['WTG ID'] == 12, 'X-coordinate.1']
vf3_distance_X_T7 = vf3_data_new.loc[vf3_data_new['WTG ID'] == 7, 'X-coordinate.1']
vf4_distance_X_T8 = vf4_data_new.loc[vf4_data_new['WTG ID'] == 8, 'X-coordinate.1']
vf5_distance_X_T9 = vf5_data_new.loc[vf5_data_new['WTG ID'] == 9, 'X-coordinate']
vf6_distance_X_T10 = vf6_data_new.loc[vf6_data_new['WTG ID'] == 10, 'X-coordinate.1']

dv_distance_Y_T1 = dv_data_new.loc[dv_data_new['WTG ID'] == 1, 'Y-coordinate']
vf1_distance_Y_T5 = vf1_data_new.loc[vf1_data_new['WTG ID'] == 5, 'Y-coordinat.1']
vf1_distance_Y_T13 = vf1_data_new.loc[vf1_data_new['WTG ID'] == 13, 'Y-coordinat.1']
vf2_distance_Y_T6 = vf2_data_new.loc[vf2_data_new['WTG ID'] == 6, 'Y-coordinate.1']
vf2_distance_Y_T11 = vf2_data_new.loc[vf2_data_new['WTG ID'] == 11, 'Y-coordinate.1']
vf2_distance_Y_T12 = vf2_data_new.loc[vf2_data_new['WTG ID'] == 12, 'Y-coordinate.1']
vf3_distance_Y_T7 = vf3_data_new.loc[vf3_data_new['WTG ID'] == 7, 'Y-coordinate.1']
vf4_distance_Y_T8 = vf4_data_new.loc[vf4_data_new['WTG ID'] == 8, 'Y-coordinate.1']
vf5_distance_Y_T9 = vf5_data_new.loc[vf5_data_new['WTG ID'] == 9, 'Y-coordinate']
vf6_distance_Y_T10 = vf6_data_new.loc[vf6_data_new['WTG ID'] == 10, 'Y-coordinate.1']

loc_DV_turbine = (6742554, 1477498)
loc_VF1_turbine_5 = (7127076, 1624359)
loc_VF1_turbine_13 = (7127062, 1624353)
loc_VF2_turbine_6 = (7125092, 1624957)
loc_VF2_turbine_11 = (7125088, 1624965)
loc_VF2_turbine_12 = (7127054, 1624358)
loc_VF3_turbine_7 = (7125500, 1624812)
loc_VF4_turbine_8 = (7128381, 1625687)
loc_VF5_turbine_9 = (7125493, 1624818)
loc_VF6_turbine_10 = (7127308, 1624078)

# ----------------------------------------------------------------
# ----------------------------------------------------------------
# ----------------------------------------------------------------

def cartesian_to_polar(turbine_loc, arr_x, arr_y):
    x_turbine, y_turbine = turbine_loc
    arr_x = np.asarray(arr_x, dtype=float)
    arr_y = np.asarray(arr_y, dtype=float)
    
    # Compute the angle in radians
    angles_rad = np.arctan2(arr_y - y_turbine, arr_x - x_turbine)
    angles_rad = -(angles_rad - np.pi / 2)  # Rotate to 0° North, clockwise
    angles_deg = np.degrees(angles_rad)
    angles_deg = (angles_deg + 360) % 360
    
    return angles_deg

# Compute angles using the cartesian_to_polar function
theta_dv_T1 = cartesian_to_polar(loc_DV_turbine, dv_distance_X_T1, dv_distance_Y_T1)
theta_VF1_T5 = cartesian_to_polar(loc_VF1_turbine_5, vf1_distance_X_T5, vf1_distance_Y_T5)
theta_VF1_T13 = cartesian_to_polar(loc_VF1_turbine_13, vf1_distance_X_T13, vf1_distance_Y_T13)
theta_VF2_T6 = cartesian_to_polar(loc_VF2_turbine_6, vf2_distance_X_T6, vf2_distance_Y_T6)
theta_VF2_T11 = cartesian_to_polar(loc_VF2_turbine_11, vf2_distance_X_T11, vf2_distance_Y_T11)
theta_VF2_T12 = cartesian_to_polar(loc_VF2_turbine_12, vf2_distance_X_T12, vf2_distance_Y_T12)
theta_VF3_T7 = cartesian_to_polar(loc_VF3_turbine_7, vf3_distance_X_T7, vf3_distance_Y_T7)
theta_VF4_T8 = cartesian_to_polar(loc_VF4_turbine_8, vf4_distance_X_T8, vf4_distance_Y_T8)
theta_VF5_T9 = cartesian_to_polar(loc_VF5_turbine_9, vf5_distance_X_T9, vf5_distance_Y_T9)
theta_VF6_T10 = cartesian_to_polar(loc_VF6_turbine_10, vf6_distance_X_T10, vf6_distance_Y_T10)
theta_monte_carlo = cartesian_to_polar(loc_DV_turbine, x_impacts, y_impacts)

theta = np.concatenate([
    theta_dv_T1,
    theta_VF1_T5, theta_VF1_T13,
    theta_VF2_T6, theta_VF2_T11, theta_VF2_T12,
    theta_VF3_T7,
    theta_VF4_T8,
    theta_VF5_T9,
    theta_VF6_T10])

# Function for distance
def euclidean_distances(turbine_loc, arr_x, arr_y):
    x_turbine, y_turbine = turbine_loc
    distances = []

    arr_x = np.asarray(arr_x)
    arr_y = np.asarray(arr_y)
    # Looping through points for every location of impact to location of turbine
    for x_val, y_val in zip(arr_x, arr_y):
        dist = np.sqrt((x_turbine - x_val)**2 + (y_turbine - y_val)**2)
        distances.append(dist)

    # Convert distances to meters from coordinates by subtracting from turbine location
    for x_val, y_val in zip(arr_x, arr_y):
        x_val = x_val - x_turbine
        y_val = y_val - y_turbine

    return np.array(distances)

point_dv_T1 = euclidean_distances(loc_DV_turbine, dv_distance_X_T1, dv_distance_Y_T1)
point_VF1_T5 = euclidean_distances(loc_VF1_turbine_5, vf1_distance_X_T5, vf1_distance_Y_T5)
point_VF1_T13 = euclidean_distances(loc_VF1_turbine_13, vf1_distance_X_T13, vf1_distance_Y_T13)
point_VF2_T6 = euclidean_distances(loc_VF2_turbine_6, vf2_distance_X_T6, vf2_distance_Y_T6)
point_VF2_T11 = euclidean_distances(loc_VF2_turbine_11, vf2_distance_X_T11, vf2_distance_Y_T11)
point_VF2_T12 = euclidean_distances(loc_VF2_turbine_12, vf2_distance_X_T12, vf2_distance_Y_T12)
point_VF3_T7 = euclidean_distances(loc_VF3_turbine_7, vf3_distance_X_T7, vf3_distance_Y_T7)
point_VF4_T8 = euclidean_distances(loc_VF4_turbine_8, vf4_distance_X_T8, vf4_distance_Y_T8)
point_VF5_T9 = euclidean_distances(loc_VF5_turbine_9, vf5_distance_X_T9, vf5_distance_Y_T9)
point_VF6_T10 = euclidean_distances(loc_VF6_turbine_10, vf6_distance_X_T10, vf6_distance_Y_T10)
point_monte_carlo = euclidean_distances(loc_DV_turbine, x_impacts, y_impacts)

points = np.concatenate([point_dv_T1,
    point_VF1_T5, point_VF1_T13,
    point_VF2_T6, point_VF2_T11, point_VF2_T12,
    point_VF3_T7,
    point_VF4_T8,
    point_VF5_T9,
    point_VF6_T10])



# ----------------------------------------------------------------
# ----------------------------------------------------------------
# ----------------------------------------------------------------

def impact_xy_meters(turbine_loc, arr_x, arr_y):
    x0, y0 = turbine_loc
    xs = np.asarray(arr_x, dtype=float)
    ys = np.asarray(arr_y, dtype=float)
    dx = xs - x0
    dy = ys - y0
    return dx, dy

dx_dv, dy_dv =         impact_xy_meters(loc_DV_turbine, dv_distance_X_T1, dv_distance_Y_T1)
dx_vf1_5, dy_vf1_5 =   impact_xy_meters(loc_VF1_turbine_5, vf1_distance_X_T5, vf1_distance_Y_T5)
dx_vf1_13, dy_vf1_13 = impact_xy_meters(loc_VF1_turbine_13, vf1_distance_X_T13, vf1_distance_Y_T13)
dx_vf2_6, dy_vf2_6 =   impact_xy_meters(loc_VF2_turbine_6, vf2_distance_X_T6, vf2_distance_Y_T6)
dx_vf2_11, dy_vf2_11 = impact_xy_meters(loc_VF2_turbine_11, vf2_distance_X_T11, vf2_distance_Y_T11)
dx_vf2_12, dy_vf2_12 = impact_xy_meters(loc_VF2_turbine_12, vf2_distance_X_T12, vf2_distance_Y_T12)
dx_vf3_7, dy_vf3_7 =   impact_xy_meters(loc_VF3_turbine_7, vf3_distance_X_T7, vf3_distance_Y_T7)
dx_vf4_8, dy_vf4_8 =   impact_xy_meters(loc_VF4_turbine_8, vf4_distance_X_T8, vf4_distance_Y_T8)
dx_vf5_9, dy_vf5_9 =   impact_xy_meters(loc_VF5_turbine_9, vf5_distance_X_T9, vf5_distance_Y_T9)
dx_vf6_10, dy_vf6_10 = impact_xy_meters(loc_VF6_turbine_10, vf6_distance_X_T10, vf6_distance_Y_T10)
x_points = np.concatenate((dx_dv, dx_vf1_5, dx_vf1_13, dx_vf2_6, dx_vf2_11, dx_vf2_12, dx_vf3_7, dx_vf4_8, dx_vf5_9, dx_vf6_10))
y_points = np.concatenate((dy_dv, dy_vf1_5, dy_vf1_13, dy_vf2_6, dy_vf2_11, dy_vf2_12, dy_vf3_7, dy_vf4_8, dy_vf5_9, dy_vf6_10))

# --------- All throws with wind speed ---------
# ---------------------- Processing results ----------------------

def process_results(ejection_angles):
    theta_obs = np.deg2rad(theta)       # to radians
    r_obs     = points                   # already in meters
    r_sim     = np.hypot(x_impacts, y_impacts)

    theta_sim = np.arctan2(y_impacts, x_impacts)
    theta_sim = -(theta_sim - np.pi / 2)  # Rotate to 0° North, clockwise
    theta_sim = (theta_sim + 2*np.pi) % (2*np.pi)
    
    percentile_sim = np.percentile(r_sim, 90)
    percentile_obs = np.percentile(r_obs, 90)

    fig, ax = plt.subplots(figsize=(8,8), subplot_kw={'projection':'polar'})
    ax.scatter(theta_sim,  r_sim,  s=10, label=f'Monte Carlo simulation with lift (N={N})', alpha=0.1, color='tab:blue')
    ax.scatter(theta_obs, r_obs, s=10, label='Observational data (530 entries)', alpha=0.5, color='tab:orange')

    ax.set_theta_zero_location('N')   # 0° at top
    ax.set_theta_direction(-1) # Clockwise rotation

    safety_distance = 1.5 * (95 + 90)
    theta_circle = np.linspace(0, 2*np.pi, 360)
    ax.plot(theta_circle,
            np.full_like(theta_circle, safety_distance),
            linestyle='--',
            color='red',
            label='Seifert safety distance with k=1.5')
    ax.plot(theta_circle,
            np.full_like(theta_circle, percentile_sim),
            linestyle='--',
            color='darkblue',
            label='$90^{th}$ percentile simulation')
    ax.plot(theta_circle,
            np.full_like(theta_circle, percentile_obs),
            linestyle='--',
            color='darkorange',
            label='$90^{th}$ percentile observations')
    
    print(f'observation percentile= {percentile_obs:.2f}m')
    print(f'lift percentile= {percentile_sim:.2f}m')
    print(f'lift: difference of 90 percentiles= {percentile_sim - percentile_obs:.2f}m')

    ax.set_rmax(safety_distance * 1.3)

    ax.set_title('Observations vs. simulation with lift')
    #ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    ax.legend(loc='lower right', bbox_to_anchor=(1.1, 0.0))
    #ax.legend(loc='best')
    plt.show()

    # ------------------------------------------------------------------
    # ----------------------- Histogram with lift ----------------------
    # ------------------------------------------------------------------


    fig2, ax2 = plt.subplots(figsize=(8, 6))
    bin_width = 10

    min_edge = np.floor(min(r_obs.min(), r_sim.min()) / bin_width) * bin_width
    max_edge = np.ceil(max(r_obs.max(), r_sim.max()) / bin_width) * bin_width

    bins = np.arange(min_edge, max_edge + bin_width, bin_width)

    ax2.hist(r_sim,
            bins=bins,
            density=True,
            label='Monte Carlo simulation with lift',
            color='tab:blue',
            edgecolor='black',
            linewidth=0.2,
            alpha=0.7,
            log=True)
    ax2.hist(r_obs,
            bins=bins,
            density=True,
            label='Observational data',
            color='tab:orange',
            edgecolor='black',
            linewidth=0.2,
            alpha=0.7,
            log=True)
    ax2.axvline(safety_distance, color='red', linestyle='--', label='safety distance with k=1.5')
    ax2.set_xlabel('Distance from turbine [m]')
    ax2.set_ylabel('Probability of ice strikes per $m^2$')
    ax2.set_title(f'Observational data (530 entries) vs. Monte Carlo simulation (N={N}) with lift')
    ax2.grid(True)
    ax2.legend()
    plt.show()

    if new_csv:
        description = f'polar_impact_locations_{N}'
        csv_file_path = f'{description}_lift.csv'
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['r [m]','theta [rad]'])
            for rs, ts in zip(r_sim, theta_sim):
                writer.writerow([rs, ts])

    # corrcoef_r = np.corrcoef(r_obs, r_sim)[0, 1]
    # print(f'{corrcoef_r}% Correlation coefficient with lift')
    # corrcoef_theta = np.corrcoef(theta_obs, theta_sim)[0, 1]
    # print(f'{corrcoef_theta}% Correlation coefficient with lift')

    # ks_stat, ks_p = ks_2samp(r_obs, r_sim)
    # theta_stat, theta_p = ks_2samp(theta_obs, theta_sim)
    # print(f"lift: KS test on theta: D={theta_stat:.3f}, p={theta_p:.3f}")
    # print(f"lift: KS test on r: D={ks_stat:.3f}, p={ks_p:.3f}")

if __name__ == '__main__':
    main()