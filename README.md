# Ice Throw Trajectories (Master’s Thesis – Milla Hjallar)

Monte Carlo simulations of ice‐throw trajectories from wind turbines. This repository contains scripts for data preprocessing, simulation, analysis, and visualization.

---

## Repository Contents

1. **`A_scatter_KDE_1m2.py`**  
   Monte Carlo simulation (gravity + drag + lift).  
   - Generates a CSV (and optionally a PNG) of scatter impact locations with KDE‐based probability densities.

2. **`AB_count_100m2.py`**  
   2D histogram of impact locations with counts per 10m x 10m.  
   - Reads the CSV from `A_scatter_KDE_1m2.py` and overlays a safety‐distance circle.

3. **`AC_probability_density_1m2.py`**  
   KDE‐based contour plot of ice‐strike probability density.  
   - Reads the same CSV and visualizes probability per 1 m² cell.

4. **`AC_probability_density_1m2_MapOverlay.py`**  
   Overlays probability contours around 74 wind turbines on a map.  
   - Uses `new_map.png` plus turbine coordinates to show contours over a real‐world map background.

5. **`NEWA_ICETHROWER_import_cleaning.py`**  
   Data preprocessing:  
   - Imports NetCDF weather data (`mesotimeseries-Point 1.nc`)  
   - Reads the Excel database (`IceThrower_dataBase_2013_2016.xlsx`)  

6. **`observations_MC_drag.py`**  
   Polar comparison (gravity and drag only) between Monte Carlo simulations and observational data.  x`  
   - Plots radial‐distance scatter and histograms (log scale).

7. **`observations_MC_lift.py`**  
   Polar comparison (with lift) between Monte Carlo simulations and observational data.  
   - Same structure as `observations_MC_drag.py`, but includes aerodynamic lift in trajectories.

8. **`parameters.ipynb`**  
   Jupyter notebook for sensitivity/parametric analyses:  
   - **Wind Speed**
   - **Mass & Area**
   - **Ejection Angle**  
   - **Rotor Radius**  
   - **Hub Height**  
   - **Initial Tangential Velocity**  
   - Generates scatter plots for each parameter set.

9. **`IceThrower_dataBase_2013_2016.xlsx`**  
   Observational ice‐throw database (2013–2016):  
   - Contains X/Y coordinates, masses [kg], lengths [cm] and widths [cm] of ice fragments

10. **`mesotimeseries-Point 1.nc`**  
    NetCDF file with weather data from NEWA (wind speed, direction, ice‐accretion times).

11. **`new_map.png`**  
    Background map used by `AC_probability_density_1m2_MapOverlay.py`.

12. **`requirements.txt`**  
    Package versions required for all scripts and the notebook.

13. **`README.md`**  
    This file.

---

## Quickstart Instructions

1. **Data Preprocessing**  
   Run `A_scatter_KDE_1m2.py` first as `AB_count_100m2.py`, `AC_probability_density_1m2.py`, `AC_probability_density_1m2_MapOverlay.py` uses data from csv generated in `A_scatter_KDE_1m2.py`.
