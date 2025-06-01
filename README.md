# Ice_throw_trajectories_masterthesis_MillaHjallar
Monte Carlo simulations of ice throw trajectories from wind turbines.

├── A_scatter_KDE_1m2.py                      # Monte Carlo simulation: drag + lift, exports CSV (and optionally PNG)
├── AB_count_100m2.py                         # 2D histogram of impact locations with safety‐distance overlay
├── AC_probability_density_1m2.py             # KDE‐based contour plot of ice‐strike probability density
├── AC_probability_density_1m2_MapOverlay.py  # Overlays probability contours on a SWEREF map with multiple turbines
├── NEWA_ICETHROWER_import_cleaning.py        # Preprocessing: import wind & ice‐fragment data, clean, compute mass/area distributions
├── observations_MC_drag.py                   # Polar comparison: Monte Carlo (without lift) vs. observational data
├── observations_MC_lift.py                   # Polar comparison: Monte Carlo (with lift) vs. observational data
├── parameters.ipynb                          # Jupyter notebook for sensitivity/parametric analyses (wind speed, mass and area, ejection angles, initial tangential velocity,                                                       radius, hub height)
├── IceThrower_dataBase_2013_2016.xlsx        # Excel file containing observational ice‐throw events (masses, dimensions, X/Y coordinates)
└── mesotimeseries-Point 1.nc                 # Weather data from NEWA
├── new_map.png                               # Background map used by MapOverlay script
├── requirements.txt                          # Exact package versions required for all code & notebook
└── README.md                                 # This file

Run A_scatter_KDE_1m2.py first as AB_count_100m2.py, AC_probability_density_1m2.py, AC_probability_density_1m2_MapOverlay.py uses data from csv generated in A_scatter_KDE_1m2.py.
