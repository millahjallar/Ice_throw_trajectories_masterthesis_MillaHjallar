import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors

RATED_WIND_SPEED = 12.0
CUT_IN_WIND_SPEED = 3.0
CUT_OUT_WIND_SPEED = 28.0
R = 45

wspd = np.linspace(0, 30, 100)

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
    
omega = [rotor_speed(ws) for ws in wspd]

plt.figure(figsize=(10, 6))
plt.plot(wspd, omega, label='Rotor speed')
plt.title('Rotor speed as a function of wind speed')
plt.xlabel('Wind speed [m/s]')
plt.ylabel('Rotor speed [rad/s]')
plt.axvline(CUT_IN_WIND_SPEED, linestyle='--', label='Cut-in wind speed', color='darkorange')
plt.axvline(RATED_WIND_SPEED, linestyle='--', label='Rated wind speed', color='green')
plt.axvline(CUT_OUT_WIND_SPEED, linestyle='--', label='Cut-out wind speed', color='red')
plt.xticks(np.arange(0, 31, 3))
plt.legend()
plt.grid()
plt.show()