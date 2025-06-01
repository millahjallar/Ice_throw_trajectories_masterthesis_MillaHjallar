import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from tqdm import tqdm
import seaborn as sns

sns.set(style='whitegrid')
np.random.seed(12345)

N = 100000                 # Number of simulations

g = 9.810                  # Gravity [m/s^2]
rho_air = 1.225            # Air density [kg/m^3]
drag_coef = 1.0            # Drag coefficient (flat-shaped ice)
mass_mean = 0.6            # Mean mass [kg]
mass_std = 0.2             # Std. dev. of mass [kg]
A_ref_mean = 0.0115        # Reference area [m^2] (mean)
hub_h = 95.0               # Hub height [m]
blade_tip_height = 140.0   # Blade tip height [m]
#R = 50.0                  # Ejection radius [m]
R = 45.0                   # Ejection radius [m]
w_hub = 7.6                # Wind speed at hub height [m/s] (mean)
shape_wind = 2.0           # Weibull shape parameter for wind
GAMDEG = 0.0               # Wind direction angle
breadth = 0.1              # Breadth of object [m]
L = 0.2                    # Length of object [m]
rho_obj = 900.0            # Density of object [kg/m^3]
omega = 1.4                # Rotor speed [rad/s]
#FIDEG = -60.0             # Azimuth angle at ejection [deg]
FIDEG = np.random.uniform(-60.0, 0.0)
WG = 0.0                   # Vertical wind (0 for now)
alpha = 1/7                # Wind shear exponent (power law)
#z_ref = hub_h             # Reference height for wind profile
#U_ref = w_hub             # Reference wind speed at hub height
k = 0.4                    # Von-Karman constant
u_star = 0.3               # Friction velocity (assumed)
z0 = 0.03                  # Roughness length (assumed)

use_log_scale = True
bin_width = 5
max_distance = 300

# scale parameter C
def weibull_scale(mean_wind_speed, shape):
    return mean_wind_speed / gamma(1.0 + 1.0/shape)

# Weibull distributed wspd
def wind_speed_distribution(scale, shape, size):
    return np.random.weibull(shape, size) * scale

# ------------ Weibull distributed wspd_hub plot ------------
scale_wind = weibull_scale(w_hub, shape_wind)
wind_speeds = wind_speed_distribution(scale_wind, shape_wind, N)

x = np.linspace(0, np.max(wind_speeds), 200)
k = shape_wind
c = scale_wind

plt.figure(figsize=(10,6))
# Weibull formula
weibull_fit = (k/c)*((x/c)**(k-1))*np.exp(-(x/c)**k)
plt.hist(wind_speeds, bins=60, density=True, label='Weibull distributed wind speed')
plt.plot(x, weibull_fit, linewidth=2, label='Weibull fitted')
plt.xlabel('Wind speed [m/s]')
plt.ylabel('Probability density')
plt.title('Weibull distributed wind speeds')
plt.legend()
plt.show()

# Normal distributed masses
def mass_distribution(mean, std, size):
    return np.abs(np.random.normal(mean, std, size))

'''# Power law for wspd at height z
def wind_profile(z):
    return U_ref * (z / z_ref)**alpha'''

# Logarithmic (log-law) wind profile
def wind_profile(z):
    z_eff = max(z, z0)
    return (u_star / k) * np.log(z_eff / z0)

# Biswas equations
def equations_of_motion(t, state, mass, C_D, A_ref):
    x, y, z, vx, vy, vz = state
    U = wind_profile(z)
    V_rel_x = vx - U
    V_rel_y = vy
    V_rel_z = vz
    V_rel = np.sqrt(V_rel_x**2 + V_rel_y**2 + V_rel_z**2)
    
    # Accelerations
    ax = -0.5 * rho_air * C_D * A_ref * V_rel_x * V_rel / mass
    ay = -0.5 * rho_air * C_D * A_ref * V_rel_y * V_rel / mass
    az = -g - 0.5 * rho_air * C_D * A_ref * V_rel_z * V_rel / mass
    
    return np.array([vx, vy, vz, ax, ay, az])

# Runge-Kutta 4th order scheme
def rk4(state0, mass, C_D, A_ref, dt=0.01, t_max=100):
    # Integrating until z=0 or t_max is reached
    t = 0.0
    state = np.array(state0)
    
    while t < t_max:
        if state[2] <= 0:  # if z <= 0, ground hit
            break
        k1 = equations_of_motion(t, state, mass, C_D, A_ref)
        k2 = equations_of_motion(t+dt/2, state+dt*k1/2, mass, C_D, A_ref)
        k3 = equations_of_motion(t+dt/2, state+dt*k2/2, mass, C_D, A_ref)
        k4 = equations_of_motion(t+dt, state+dt*k3, mass, C_D, A_ref)
        
        state = state + dt*(k1+(2*k2)+(2*k3)+k4)/6.0
        t += dt
    
    return state
print('done with rk4')

def simulate_trajectory(mass, C_D, A_ref):
    phi = np.deg2rad(FIDEG) # Ejection angle
    V_t = omega * R # Blade tip speed
    
    # Initial position of turbine
    x0 = 0.0
    y0 = 0.0
    z0 = blade_tip_height
    
    # Initial velocities:
    # vx0, vy0 determined by blade tangential velocity.
    vx0 = -V_t * np.sin(phi)
    vy0 = V_t * np.cos(phi)
    vz0 = 0.0  # no vertical initial velocity component
    
    state0 = [x0, y0, z0, vx0, vy0, vz0]
    
    # dt=0.01
    final_state = rk4(state0, mass, C_D, A_ref, dt=0.1, t_max=100)
    x_impact = final_state[0]
    y_impact = final_state[1]
    return x_impact, y_impact

# Weibull distributed wspd and normal dist. mass
mean_wind_speed = w_hub
scale_wind = weibull_scale(mean_wind_speed, shape_wind)
masses = mass_distribution(mass_mean, mass_std, N)

A_ref = A_ref_mean

def rotor_speed(wind_speed, max_omega=1.4, max_wind=20.0):
    return max_omega * np.clip(wind_speed / max_wind, 0, 1)

def delta_omega(mass, C_D, A_ref, wind_speed):
    # Determining rotor speed from wind speed
    omega_dynamic = rotor_speed(wind_speed)
    
    phi = np.deg2rad(FIDEG)
    V_t = omega_dynamic * R
    
    # Initial conditions
    x0, y0 = 0.0, 0.0
    z0 = blade_tip_height  # Always starts at the blade tip height
    vx0 = -V_t * np.sin(phi)
    vy0 = V_t * np.cos(phi)
    vz0 = 0.0  # No initial vertical velocity
    state0 = [x0, y0, z0, vx0, vy0, vz0]
    
    # Integrating trajectory
    #final_state = rk4(state0, mass, C_D, A_ref, dt=0.01, t_max=100)
    final_state = rk4(state0, mass, C_D, A_ref, dt=0.1, t_max=100)
    x_impact, y_impact = final_state[0], final_state[1]
    return x_impact, y_impact

print('Starting with Monte Carlo')
# Monte carlo simulation
impact_distances = np.zeros(N)
for i in tqdm(range(N)):
    # Generating random wind speed for each simulation
    wind_speed = wind_speed_distribution(scale_wind, shape_wind, 1)[0]
    mass = masses[i]
    x_impact, y_impact = delta_omega(mass, drag_coef, A_ref, wind_speed)
    distance = np.sqrt(x_impact**2 + y_impact**2)
    impact_distances[i] = distance
print('Monte carlo done')

bin_edges = np.arange(0, max_distance+bin_width, bin_width)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
counts, _ = np.histogram(impact_distances, bins=bin_edges)

areas = np.pi * (bin_edges[1:]**2 - bin_edges[:-1]**2)
strikes_per_m2 = counts / areas
probability_per_m2 = strikes_per_m2 / N

plt.figure(figsize=(10, 6))
#plt.plot(bin_centers, probability_per_m2, 'o-', label='Probability per $m^2$')
plt.plot(bin_centers, probability_per_m2, label='Probability per $m^2$')

if use_log_scale:
    plt.yscale('log')

plt.xlabel('Distance from turbine [m]')
plt.ylabel('Probability of ice strikes per $m^2$')
plt.title('Probability of ice strikes vs. distance')
plt.grid(True)
plt.legend()
plt.show()