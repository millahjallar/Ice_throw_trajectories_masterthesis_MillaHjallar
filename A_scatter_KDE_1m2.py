import numpy as np
np.random.seed(12345)
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
import datetime
import scipy.stats as st
import matplotlib.colors as colors


N = 100000                        # number of simulations

'''
new_csv                      : Set to False if you don't want to create a new csv file.
new_png                      : Set to True if you want to create a new png file.
'''

new_csv = True
new_png = False

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

# ------------------------------------------------------------------------------------------
# ---------------------- Importing necessary data for windformation;) ----------------------
# ------------------------------------------------------------------------------------------

from NEWA_ICETHROWER_import_cleaning import wspd, wdir

ws = np.array(wspd)
wd = np.array(wdir)

idx_wswd = np.random.choice(len(ws), size=N, replace=True)
WSPD = ws[idx_wswd]
WDIR = wd[idx_wswd]

# ------------------------------------------------------------------------------------------
# ---------------------- Importing necessary data for area and masses ----------------------
# ------------------------------------------------------------------------------------------

from NEWA_ICETHROWER_import_cleaning import lengths0, widths0, masses0

areas = []
masses = []

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

# ---------------------- Defining functions ----------------------
# Logarithmic (log-law) wind profile
def wind_profile(z, u_star):
    z_eff = max(z, z0)
    return (u_star / VON_K) * np.log(z_eff / z0)

# Friction velocity
def U_star(wspd):
    return (VON_K * wspd) / np.log(HUB_H / z0)

# Rotor speed
def rotor_speed(wind_speed):
    MAX_OMEGA_ROTOR_SPEED = (7 * RATED_WIND_SPEED) / R 
    
    if wind_speed < CUT_IN_WIND_SPEED:
        return 0.0
    elif wind_speed < RATED_WIND_SPEED:
        return ((wind_speed - CUT_IN_WIND_SPEED) / (RATED_WIND_SPEED - CUT_IN_WIND_SPEED)) * MAX_OMEGA_ROTOR_SPEED
    elif wind_speed <= CUT_OUT_WIND_SPEED:
        return MAX_OMEGA_ROTOR_SPEED
    else:
        return 0.0

# Decomposing wind speed into components
def wind_components(wspd, wdir):
    theta = np.radians(wdir)
    '''
    theta = 0deg:  wspd = (0, -wspd)
    theta = 90deg: wspd = (-wspd, 0)
    '''
    wx = -wspd * np.sin(theta)
    wy = -wspd * np.cos(theta) 
    return wx, wy

# Equations of motion determining the trajectory of the ice fragments
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

# Runge-Kutta 4th order scheme solving the equations of motion
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

# Initial conditions
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
    x, y, z, vx, vy, vz = final_state

    v = np.sqrt(vx**2 + vy**2 + vz**2)  # impact velocity

    return x, y, v

x_impacts = np.zeros(N)
y_impacts = np.zeros(N)
v_impacts = np.zeros(N)

def main():
    random_angles = np.random.uniform(0, 360, N)
    ejection_angles = np.zeros(N)
    omegas = np.zeros(N)
    u_stars = np.zeros(N)

    # Monte Carlo simulation
    for i in tqdm(range(N)):
        angle = random_angles[i]
        ejection_angles[i] = angle
        wspd_i = WSPD[i]
        wdir_i = WDIR[i]
        u_star_i = U_star(wspd_i)
        u_stars[i] = u_star_i
        omega_i = rotor_speed(wspd_i)
        omegas[i] = omega_i

        x_impacts[i], y_impacts[i], v_impacts[i] = trajectory(
            omega_i,
            mass_distribution[i],
            angle,
            area_distribution[i],
            wspd_i,
            wdir_i,
            u_star_i
        )
    
    process_results(ejection_angles, omegas, u_stars)

# Plotting
def process_results(ejection_angles, omegas, u_stars):
    positions = np.vstack((x_impacts, y_impacts))
    kde = st.gaussian_kde(positions)
    probability_density = kde(positions)

    description = 'Ice_thrower_simulation'
    timestamp = datetime.date.today().strftime('%d.%m.%Y')

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        x_impacts, y_impacts,
        c=probability_density, s=1, cmap='Blues', norm=colors.LogNorm(),
        label='Impact locations'
    )
    plt.colorbar(scatter, label='Probability of ice strikes per m$^{2}$')
    plt.scatter(0, 0, s=20, color='red', label='Initial position of ice throw')
    plt.xlabel('X-coordinate of impact location [m]')
    plt.ylabel('Y-coordinate of impact location [m]')
    plt.title('Impact locations of ice throws in xy-plane')
    plt.legend(loc='upper right')
    plt.grid(True)
    if new_png:
        plt.savefig(f'plot_{timestamp}_{N}.png')
    plt.show()

    print(f'max x: {np.max(np.abs(x_impacts))}', f'\nmax y: {np.max(np.abs(y_impacts))}')

    # Saving results to CSV
    if new_csv:
        csv_file_path = f'{description}_{N}its_{timestamp}.csv'
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['wind speed [m/s]', 'wind direction [deg]', 'X-coordinate [m]', 'Y-coordinate [m]', 'mass [kg]',
                             'Ejection angle [deg]', 'Frontal area of ice [m^2]', 'Probability [m^-2]', 'omega [rad/s]', 'Impact velocity [m/s]', 'Friction velocity [m/s]'])
            for ws, wd, x, y, m, F, A, P, om, v, us in zip(
                WSPD, WDIR, x_impacts, y_impacts, mass_distribution, ejection_angles, area_distribution, probability_density, omegas, v_impacts, u_stars
            ):
                writer.writerow([ws, wd, x, y, m, F, A, P, om, v, us])

if __name__ == '__main__':
    main()

