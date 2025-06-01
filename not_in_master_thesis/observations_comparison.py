import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_drag = pd.read_csv('polar_impact_locations_10000_drag.csv')
df_lift = pd.read_csv('polar_impact_locations_10000_lift.csv')

# ------------------------------------------------------
# 'r [m]','theta [rad]'
r_drag = df_drag['r [m]']
theta_drag = df_drag['theta [rad]']

r_lift = df_lift['r [m]']
theta_lift = df_lift['theta [rad]']

# ------------------------------------------------------

percentile_drag = np.percentile(r_drag, 90)
percentile_lift = np.percentile(r_lift, 90)

fig, ax = plt.subplots(figsize=(8,8), subplot_kw={'projection':'polar'})
ax.scatter(theta_drag, r_drag, s=10, label='Monte Carlo simulation without lift', alpha=0.2, color='tab:green')
ax.scatter(theta_lift, r_lift, s=10, label='Monte Carlo simulation without lift', alpha=0.2, color='tab:blue')

safety_distance = 1.5 * (95 + 90)
theta_circle = np.linspace(0, 2*np.pi, 360)

ax.plot(theta_circle,
        np.full_like(theta_circle, safety_distance),
        linestyle='--',
        color='red',
        label='Seifert safety distance with k=1.5')
# plotting percentiles as circles
ax.plot(theta_circle,
        np.full_like(theta_circle, percentile_drag),
        linestyle='--',
        color='forestgreen',
        label='90% percentile without lift simulation')
ax.plot(theta_circle,
        np.full_like(theta_circle, percentile_lift),
        linestyle='--',
        color='darkblue',
        label='90% percentile with lift simulation')

print(f'drag and seifert: difference of 90 percentiles= {safety_distance - percentile_drag:.2f}m')
print(f'lift and seifert: difference of 90 percentiles= {safety_distance - percentile_lift:.2f}m')
print(f'lift vs. drag: difference of 90 percentiles= {percentile_lift - percentile_drag:.2f}m')

ax.set_title('Observations vs. simulation without lift')
#ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
ax.legend(loc='lower right', bbox_to_anchor=(1.1, 0.0))
plt.show()