import helper_functions as hf
import numpy as np
import matplotlib.pyplot as plt

def plot_dep_vars(folder_path):
    
    dep_vars = np.genfromtxt(folder_path + "/dependent_variable_history.dat").T
    
    time_in_days = dep_vars[0] / (24*60**2)
    
    vx_ground = dep_vars[2]
    vy_ground = dep_vars[3]
    vz_ground = dep_vars[4]
    
    geo_lat = dep_vars[5] * np.rad2deg(1)
    lat = dep_vars[6] * np.rad2deg(1)
    long = dep_vars[7] * np.rad2deg(1)
    
    hf.plot_arrays(time_in_days, [vx_ground, vy_ground, vz_ground], keep_in_memory=True, 
                   y_label="Goundspeed [m/s]", x_label="Time [days]", legend=["Vx", "Vy", "Vz"])
    
    hf.plot_arrays(time_in_days, [geo_lat, lat, long], keep_in_memory=True, 
                   y_label="lat/long [degrees]", x_label="Time [days]", legend=["geo lat", "lat", "long"])
    
    
    plt.show()
    
    
plot_dep_vars(hf.sim_data_dir + "/integrator_analysis/benchmarks/rkf_45")