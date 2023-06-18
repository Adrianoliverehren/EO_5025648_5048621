import numpy as np
from tudatpy.kernel.astro import frame_conversion
import os
import json
from tudatpy.io import save2txt
from tudatpy.kernel.math import geometry
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
import sys
import pathlib
import matplotlib.colors as mcolors
from matplotlib import colors
from mpl_toolkits.basemap import Basemap


def remove_folder_from_path(path : str, no_to_remove=1):
    path = path.replace("\\", "/")
    if no_to_remove >= path.count('/'):
        path = '/'
    else:
        for _ in range(no_to_remove):
            path = os.path.dirname(path)    
    return path

root_dir = os.path.dirname(__file__)
report_dir = remove_folder_from_path(os.path.dirname(__file__), 2) + "/overleaf"
sim_data_dir = os.path.dirname(__file__) + "/SimulationData"
plots_dir = os.path.dirname(__file__) + "/SimulationData/plots/"


def plot_arrays(
    x_arrays, y_arrays, path_to_save=False, title=None, x_label=None, y_label=None, scale_factor=None,
    legend=None, grid=True, x_log=False, linestyles=None, linewiths=None, plot_size=[4,4], colors=None, x_lim=None, legend_pos=None,
    force_xticks=False, force_sci_notation=False, custom_legend_entries=None, custom_markings=None, markers=None, marker_colors=None,
    markerfacecolors=None, marker_sizes=None, keep_in_memory=False, y_log=False, markings=None, additional_save_path=None, **kwargs):
    
    if type(x_arrays[0]) not in [list, np.ndarray]:
        x_arrays = [x_arrays] * len(y_arrays)
    
    plt.figure()
    
    if plot_size:
        fig = plt.gcf()
        fig.set_size_inches(plot_size[0], plot_size[1])
    
    if scale_factor:
        y_arrays = y_arrays*scale_factor
    
    if title:
        plt.title(title)
        
    if x_label:
        plt.xlabel(x_label)

    if y_label:
        plt.ylabel(y_label)
    
    i = 0
    for x_array, y_array in zip(x_arrays, y_arrays):
        line_plot, = plt.plot(x_array, y_array)
        if linestyles:
            plt.setp(line_plot, linestyle=linestyles[i])
        if linewiths:
            plt.setp(line_plot, linewidth=linewiths[i])            
        if colors:
            if len(colors) == 1:
                colors = colors * len(y_arrays)
            plt.setp(line_plot, color=colors[i])
        if markings:
            if not type(markings) == list:
                markings = ["."] * len(y_arrays)
            else:
                if len(markings) == 1:
                    markings = markings * len(y_arrays)
            plt.setp(line_plot, marker=markings[i])
            
        i+=1 
        
    if custom_markings:
        if markerfacecolors == None:
            markerfacecolor=['none'] * len(custom_markings)
        elif len(markerfacecolors) == 1:
            markerfacecolors = markerfacecolors * len(custom_markings)
            
        if markers == None:
            markers = ["o"] * len(custom_markings)
        elif len(markers) == 1:
            markers = markers * len(custom_markings)
            
        if marker_colors == None:
            marker_colors = ["black"] * len(custom_markings)
        elif len(marker_colors) == 1:
            marker_colors = marker_colors * len(custom_markings)
        
        if marker_sizes == None:
            marker_sizes = [5] * len(custom_markings)
        elif len(marker_sizes) == 1:
            marker_sizes = marker_sizes * len(custom_markings)
            
        for i, mark in enumerate(custom_markings):
            plt.plot(mark[0], mark[1], marker=markers[i], markerfacecolor=markerfacecolor[i], color=marker_colors[i],
                     markersize=marker_sizes[i], markeredgewidth=2)

    
    if len(kwargs) > 0:
        
        if "label_points" in kwargs:
            
            for annotation in kwargs["label_points"]:
            
                text = annotation["text"]
                x, y = annotation["x"], annotation["y"]
                x_text, y_text = annotation["x_text"], annotation["y_text"]
                
                
                plt.annotate(text, xy=(x, y), xytext=(x_text,y_text), 
                textcoords='offset points', ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', 
                                color='red'))
        
    
    if force_sci_notation:
        if force_sci_notation == "x":
            plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        if force_sci_notation == "y":
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        
    if legend:
        if custom_legend_entries:
            plt.legend(custom_legend_entries, legend)
        else:
            plt.legend(legend)
        if legend_pos:
            if legend_pos == "below":
                plt.legend(legend, loc='upper center', bbox_to_anchor=(0.5, -0.18))
            if legend_pos == "right":
                plt.legend(legend, loc='center left', bbox_to_anchor=(1, 0.5))
            if legend_pos == "inside_top_left":
                plt.legend(legend, loc='upper left')
                
                
        
    if grid:
        plt.grid()
        
    if x_log:
        plt.xscale('log')
        
    if y_log:
        plt.yscale('log')
        
    if x_lim:
        plt.xlim(x_lim[0], x_lim[1])
    
    if force_xticks:
        plt.xticks(force_xticks)
    
    plt.tight_layout()
        
    if path_to_save:        
        path = pathlib.Path(path_to_save)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_to_save, bbox_inches='tight')
        if additional_save_path:
            path = pathlib.Path(additional_save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(additional_save_path, bbox_inches='tight')

    if keep_in_memory == False:
        plt.close()

def plot_heatmap_scatter(x_array, y_array, z_array, path_to_save=False, title=None, x_label=None, y_label=None,
    grid=True, x_log=False, y_log=False, plot_size=[4,4], additional_save_path=None, keep_in_memory=False, 
    normalize_colourbar=None, z_label=None, filter_ids=None ,**kwargs):
    
    plt.figure()
    
    
    if plot_size:
        fig = plt.gcf()
        fig.set_size_inches(plot_size[0], plot_size[1])
    
    if title:
        plt.title(title)
        
    if x_label:
        plt.xlabel(x_label)

    if y_label:
        plt.ylabel(y_label)    
    orig_map=plt.cm.get_cmap('jet')
  
    # reversing the original colormap using reversed() function
    reversed_map = orig_map.reversed()
        
    if filter_ids:
        
        x_unwanted = np.take(x_array, filter_ids)
        y_unwanted = np.take(y_array, filter_ids)
        z_unwanted = np.take(z_array, filter_ids)
        
        plt.scatter(x_unwanted, y_unwanted, c="gray", marker=".")
        
        x_array = np.delete(x_array, filter_ids)
        y_array = np.delete(y_array, filter_ids)
        z_array = np.delete(z_array, filter_ids)
        

    if normalize_colourbar=="start_at_zero":
        
        ids_below_zero = np.argwhere(z_array < 0).T[0]
        ids_above_zero = np.argwhere(z_array >= 0).T[0]
        
        
        x_below_zero = np.take(x_array, ids_below_zero)
        y_below_zero = np.take(y_array, ids_below_zero)
        z_below_zero = np.take(z_array, ids_below_zero)
        
        plt.scatter(x_below_zero, y_below_zero, c="gray", marker=".")
        x_array = np.take(x_array, ids_above_zero)
        y_array = np.take(y_array, ids_above_zero)
        z_array = np.take(z_array, ids_above_zero)
    
        plt.scatter(x_array, y_array, c=z_array, cmap=reversed_map)
    
    if normalize_colourbar=="log":
        
        
        ids_non_zero = np.argwhere(z_array > 1e-15).T[0]
        x_array = np.take(x_array, ids_non_zero)
        y_array = np.take(y_array, ids_non_zero)
        z_array = np.take(z_array, ids_non_zero)
        
        # sys.exit()
        norm=colors.LogNorm(vmin=np.min(z_array), 
        vmax=np.max(z_array))
        
        plt.scatter(x_array, y_array, c=z_array, cmap=reversed_map, norm=norm)
    
    else:
        
        plt.scatter(x_array, y_array, c=z_array, cmap=reversed_map)
    

    # Add a colorbar for reference
    colorbar = plt.colorbar()
    if z_label:
        colorbar.set_label(z_label)

    # Set labels and title
    
    if grid:
        plt.grid()
        
    if x_log:
        plt.xscale("log")
    if y_log:
        plt.yscale("log")
        
    plt.tight_layout()
        
    if path_to_save:        
        path = pathlib.Path(path_to_save)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_to_save, bbox_inches='tight')
        if additional_save_path:
            path = pathlib.Path(additional_save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(additional_save_path, bbox_inches='tight')

    if keep_in_memory == False:
        plt.close()

def plot_heatmap_scatter_multi_decision_vars(x_arrays, y_arrays, z_arrays, markers, path_to_save=False, title=None, x_label=None, y_label=None,
    grid=True, x_log=False, y_log=False, plot_size=[4,4], additional_save_path=None, keep_in_memory=False, 
    normalize_colourbar=None, z_label=None, filter_ids=None, legend=None, marker_sizes=None,**kwargs):
    
    plt.figure()
    
    
    if plot_size:
        fig = plt.gcf()
        fig.set_size_inches(plot_size[0], plot_size[1])
    
    if title:
        plt.title(title)
        
    if x_label:
        plt.xlabel(x_label)

    if y_label:
        plt.ylabel(y_label)   
    orig_map=plt.cm.get_cmap('jet')
  
    # reversing the original colormap using reversed() function
    reversed_map = orig_map.reversed()
    i = 0
    for x_array, y_array, z_array in zip(x_arrays, y_arrays, z_arrays):
        
        if filter_ids:
            
            x_unwanted = np.take(x_array, filter_ids)
            y_unwanted = np.take(y_array, filter_ids)
            z_unwanted = np.take(z_array, filter_ids)
            
            plt.scatter(x_unwanted, y_unwanted, c="gray", marker=markers[i], s=marker_sizes[i])
            
            x_array = np.delete(x_array, filter_ids)
            y_array = np.delete(y_array, filter_ids)
            z_array = np.delete(z_array, filter_ids)
            

        if normalize_colourbar=="start_at_zero":
            
            ids_below_zero = np.argwhere(z_array < 0).T[0]
            ids_above_zero = np.argwhere(z_array >= 0).T[0]
            
            
            x_below_zero = np.take(x_array, ids_below_zero)
            y_below_zero = np.take(y_array, ids_below_zero)
            z_below_zero = np.take(z_array, ids_below_zero)
            
            plt.scatter(x_below_zero, y_below_zero, c="gray", marker=markers[i], s=marker_sizes[i])
            x_array = np.take(x_array, ids_above_zero)
            y_array = np.take(y_array, ids_above_zero)
            z_array = np.take(z_array, ids_above_zero)
        
            plt.scatter(x_array, y_array, c=z_array, cmap=reversed_map, marker=markers[i], s=marker_sizes[i])
        
        if normalize_colourbar=="log":
            
            
            ids_non_zero = np.argwhere(z_array > 1e-15).T[0]
            x_array = np.take(x_array, ids_non_zero)
            y_array = np.take(y_array, ids_non_zero)
            z_array = np.take(z_array, ids_non_zero)
            
            # sys.exit()
            norm=colors.LogNorm(vmin=np.min(z_array), 
            vmax=np.max(z_array))
            
            plt.scatter(x_array, y_array, c=z_array, cmap=reversed_map, norm=norm, marker=markers[i], s=marker_sizes[i])
        
        else:
            
            plt.scatter(x_array, y_array, c=z_array, cmap=reversed_map, marker=markers[i], s=marker_sizes[i])
        i += 1
        
    # Add a colorbar for reference
    colorbar = plt.colorbar()
    if z_label:
        colorbar.set_label(z_label)

    # Set labels and title
    
    if legend:
        from matplotlib.lines import Line2D

        mod_sizes = [2,4,6,8]


        legend_elements = [Line2D([0], [0], marker=markers[i], color='black', label=legend[i], markerfacecolor='black', markersize=mod_sizes[i], lw=0) for i in range(len(legend))]
                
        plt.legend(handles=legend_elements)
    
    
    if grid:
        plt.grid()
        
    if x_log:
        plt.xscale("log")
    if y_log:
        plt.yscale("log")
        
    plt.tight_layout()
        
    if path_to_save:        
        path = pathlib.Path(path_to_save)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_to_save, bbox_inches='tight')
        if additional_save_path:
            path = pathlib.Path(additional_save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(additional_save_path, bbox_inches='tight')

    if keep_in_memory == False:
        plt.close()

def plot_lat_long(longitude, latitude, linewiths=None, colors=None, keep_in_memory=False, 
                  path_to_save=False, additional_save_path=None):
    
    plt.figure()
    
    m = Basemap(projection='mill', lon_0=180)

    # Draw coastlines and fill the continents
    m.drawcoastlines()
    m.fillcontinents(color='lightgray')

    # Draw parallels and meridians
    m.drawparallels(range(-90, 91, 30), labels=[True, False, False, False])
    m.drawmeridians(range(-180, 181, 60), labels=[False, False, False, True])
        
    if type(longitude[0]) not in [list, np.ndarray]:
        longitude = [longitude] * len(latitude)
        
    i = 0
    for x_array, y_array in zip(longitude, latitude):
        x_array[x_array < 0] += 360
        x, y = m(x_array, y_array)
        line_plot, = m.plot(x, y)
        if linewiths:
            plt.setp(line_plot, linewidth=linewiths[i])            
        if colors:
            if len(colors) == 1:
                colors = colors * len(latitude)
            plt.setp(line_plot, color=colors[i])
        i +=1
    
    
    
    plt.tight_layout()
        
    if path_to_save:        
        path = pathlib.Path(path_to_save)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_to_save, bbox_inches='tight')
        if additional_save_path:
            path = pathlib.Path(additional_save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(additional_save_path, bbox_inches='tight')

    if keep_in_memory == False:
        plt.close()
    
    pass

def calculate_constraint_objective_functions(num_states_dic, dep_vars_dic, decision_vars):
    """_summary_

    Args:
        dep_vars_dic (dict): _description_
        decision_vars (dict): _description_
    """
    
    capsule = geometry.Capsule(
        decision_vars["nose_radius"],
        decision_vars["mid_radius"],
        decision_vars["rear_len"],
        decision_vars["rear_angle"],
        decision_vars["side_radius"],
    )
    
    
    time_array = np.array(list(dep_vars_dic.keys()))
    
    num_states = np.array(list(num_states_dic.values())).T
    
    dep_vars = np.array(list(dep_vars_dic.values())).T
    
    ########################################################
    # heat
    
    rho = dep_vars[24]
    aoa = dep_vars[8]
    V_air = np.linalg.norm(dep_vars[15:18], axis=0)
    k = 1.83e-4 / np.sqrt(decision_vars["nose_radius"])
    stagnation_heat_flux = k * rho**0.5 * V_air**3 # Chapman
    M = dep_vars[0]
    Rs_Rm = decision_vars["side_radius"]/decision_vars["mid_radius"]
    theta = decision_vars["rear_angle"]
    
    theta = np.arcsin((decision_vars["mid_radius"] - decision_vars["side_radius"] )/(decision_vars["nose_radius"] - decision_vars["side_radius"]))
    
    c1, c2, c3, c4, c5 = -0.0006, 0.0185, -0.5321, -0.2939, 1.3630
    
    max_heat_flux = (c1*M + c2*np.deg2rad(aoa) + c3*Rs_Rm + c4*np.deg2rad(theta)+ c5) * stagnation_heat_flux
    peak_max_heat_flux = np.max(max_heat_flux)
    total_heat_load = np.trapz(max_heat_flux, time_array)
    
    ########################################################
    # range 
    
    V_wrt_ground = dep_vars[11:14]
    
    pos_vector = num_states[0:3]
    unit_pos_vector = pos_vector / np.linalg.norm(pos_vector, axis=0)
    
    V_perpendicular_to_ground = np.array([Vi.dot(u_vec) * u_vec for Vi, u_vec in zip(V_wrt_ground.T, unit_pos_vector.T)]).T
    
    V_parallel_to_ground = V_wrt_ground - V_perpendicular_to_ground
    
    unit_V_start_parallel_to_ground = V_parallel_to_ground.T[0] / np.linalg.norm(V_parallel_to_ground.T[0])
    
    V_parallel_to_start = np.array([Vi.dot(unit_V_start_parallel_to_ground) * unit_V_start_parallel_to_ground for Vi in V_parallel_to_ground.T]).T
    
    V_perpendicular_to_start = V_parallel_to_ground - V_parallel_to_start
    
    
    cummulative_along_track_dist = cumtrapz(np.linalg.norm(V_parallel_to_start, axis=0), time_array, initial=0)
    cummulative_cross_track_dist = cumtrapz(np.linalg.norm(V_perpendicular_to_start, axis=0), time_array, initial=0)
    along_track_dist = cummulative_along_track_dist[-1]
    cross_track_dist = cummulative_cross_track_dist[-1]
        
    
    initial_rsw_rotation_frame = frame_conversion.inertial_to_rsw_rotation_matrix(num_states.T[0])

    rsw_vel = np.array([initial_rsw_rotation_frame @ vi for vi in num_states[3:6].T])

    along_track_vel = rsw_vel.T[1]
    cross_track_vel = rsw_vel.T[2]
    
    
    cummulative_along_track_dist = cumtrapz(along_track_vel, time_array, initial=0)
    cummulative_cross_track_dist = cumtrapz(cross_track_vel, time_array, initial=0)
    
    rsw_along_track_dist = cummulative_along_track_dist[-1]
    rsw_cross_track_dist = cummulative_cross_track_dist[-1]
    
    
    ########################################################
    # force integrations
    
    side_force = dep_vars[2] * capsule.middle_radius**2 * np.pi * np.sin(dep_vars[10]) * dep_vars[18]
    
    lift_force = dep_vars[2] * capsule.middle_radius**2 * np.pi * dep_vars[18]
    
    cummulative_side_force = cumtrapz(side_force, time_array, initial=0)
    cummulative_lift_force = cumtrapz(lift_force, time_array, initial=0)
    
    
    peak_g_load = np.max(dep_vars[14] / 9.80665)
    
    ########################################################
    # skip altitude 
    dh_dt = np.diff(dep_vars[1])/np.diff(time_array)
    
    altitude = dep_vars[1]
    
    skip_altitudes = []
    
    for i in range(len(dh_dt)):
        if dh_dt[i] > 0:
            # skip is occuring
            skip_altitudes.append(altitude[i])
    
    if len(skip_altitudes) == 0:
        # no skip occurs
        skip_altitudes = [0]
        
    skip_altitudes = np.array(skip_altitudes)
    
    max_skip_altitude = float(np.max(skip_altitudes))
    
    ########################################################
    # capsule volume
    
    vol = capsule.volume
    
    # capsule surface area modelled as cone
    
    Rm = decision_vars["mid_radius"]
    Rs = decision_vars["side_radius"]
    RN = decision_vars["nose_radius"]
    Lc = decision_vars["rear_len"]
    
    t_sp1max = np.arcsin((Rm - Rs) / (RN - Rs))
    t_tmin = t_sp1max
    t_tmax = decision_vars["rear_angle"]
    t_c = decision_vars["rear_angle"]
    
    Rc1 = Rm * (1 - np.cos(t_c))
    Rsp2 = (Rc1 - Lc * np.tan(t_c)) / np.cos(t_c)
    
    Lt = Rs * (np.sin(t_tmin) + np.sin(t_tmax))
    Lsp1 = RN * (1 - np.sin(t_sp1max))
    Lsp2 = Rsp2 * (1 - np.sin(t_c))
    
    L = Lc + Lt + Lsp1 + Lsp2
    
    area = np.pi * Rm * (Rm + np.sqrt(L**2 + Rm**2))
    
    vol_efficiency = 6 * np.sqrt(np.pi) * (vol / area**(3/2))
    
    ########################################################
    # stability?
    
    dalpha_dt = np.insert(np.diff(aoa)/np.diff(time_array), 0, 0)
    
    Cmx = dep_vars[19]
    Cmy = dep_vars[20]
    Cmz = dep_vars[21]
    
    # dCmx_dt = np.insert(np.diff(Cmx)/np.diff(time_array), 0, 0)
    dCmy_dt = np.insert(np.diff(Cmy)/np.diff(time_array), 0, 0)
    # dCmz_dt = np.insert(np.diff(Cmz)/np.diff(time_array), 0, 0)
    
    dCmy_daoa = []
    aoa_chage_time_array = []
    
    for i in range(len(dalpha_dt)):
        if dalpha_dt[i] !=0:
            dCmy_daoa.append(dCmy_dt[i] / dalpha_dt[i])
            aoa_chage_time_array.append(time_array[i])
            pass
        
    if len(dCmy_daoa) == 0:
        dCmy_daoa = [0]
        aoa_chage_time_array = [0]
        
    max_dCmy_daoa = max(dCmy_daoa)
    
    out = {
        "peak_heat_flux": peak_max_heat_flux,
        "total_heat_load": total_heat_load,
        "along_track_dist": along_track_dist,
        "cross_track_dist": cross_track_dist,
        "rsw_along_track_dist": rsw_along_track_dist,
        "rsw_cross_track_dist": rsw_cross_track_dist,
        "peak_g_load": peak_g_load,
        "vol_efficiency": vol_efficiency,
        "max_skip_altitude": max_skip_altitude,
        "dCmy_daoa": dCmy_daoa,
        "aoa_chage_time_array": aoa_chage_time_array,
        "max_dCmy_daoa": max_dCmy_daoa,
        "final_altitude": altitude[-1],
        "final_latitude": dep_vars[22][-1],
        "final_longitude": dep_vars[23][-1],
        "cummulative_side_force": cummulative_side_force[-1],
        "cummulative_lift_force": cummulative_lift_force[-1],
    }
    
    return out



def determine_if_constraints_are_violated(objective_vals_dict):
    
    # check if altitude is close to final altitude
    
    final_altitude_violation = False
    g_load_violation = False
    dCmy_daoa_violation = False
    peak_flux_violation = False
    max_skip_altitude = False
    
    distance_dic = {}
    
    if not 24000 < objective_vals_dict["final_altitude"] < 26000:
        final_altitude_violation = True
        
    # check G load constraint
    
    distance_dic["g_load_distance"] = 10 - objective_vals_dict["peak_g_load"]
    if objective_vals_dict["peak_g_load"] > 10:
        g_load_violation = True
        
    # check stability
    
    distance_dic["dCmy_daoa_distance"] = 0 - objective_vals_dict["max_dCmy_daoa"]
    if objective_vals_dict["max_dCmy_daoa"] > 0:
        dCmy_daoa_violation = True
        
    #check peak heat flux
    distance_dic["peak_flux_distance"] = 800000 - objective_vals_dict["peak_heat_flux"]
    if objective_vals_dict["peak_heat_flux"] > 800000:
        peak_flux_violation = True
        
    #max skip altitude
    distance_dic["max_skip_altitude_distance"] = 120000 - objective_vals_dict["max_skip_altitude"]
    if objective_vals_dict["max_skip_altitude"] > 120000:
        max_skip_altitude = True
        
    pass_fail_lst = [final_altitude_violation, g_load_violation, dCmy_daoa_violation, peak_flux_violation, max_skip_altitude]
    
    
    return pass_fail_lst, distance_dic


def find_footprint_area_for_multiple_trajectories(folder, no_trajectories):
    
    no_rejected = 0
    
    feasible_cross_range_array = []
    feasible_down_range_array = []
    feasible_rsw_cross_range_array = []
    feasible_rsw_down_range_array = []
    feasible_latitude_array = []
    feasible_longitude_array = []
    unfeasible_cross_range_array = []
    unfeasible_down_range_array = []
    unfeasible_rsw_cross_range_array = []
    unfeasible_rsw_down_range_array = []
    unfeasible_latitude_array = []
    unfeasible_longitude_array = []
    good_trajectory_ids = []
    
    for i in range(no_trajectories):
        
        objective_vals = create_dic_drom_json(folder + f"/trajectory_{i}/objectives_and_constraints.dat")
        
        violations, distance_dic = determine_if_constraints_are_violated(objective_vals)
    
        
        if not any(violations):
            good_trajectory_ids.append(i)
            feasible_cross_range_array.append(objective_vals["cross_track_dist"])
            feasible_down_range_array.append(objective_vals["along_track_dist"])
            feasible_rsw_cross_range_array.append(objective_vals["rsw_cross_track_dist"])
            feasible_rsw_down_range_array.append(objective_vals["rsw_along_track_dist"])
            feasible_latitude_array.append(objective_vals["final_latitude"])
            feasible_longitude_array.append(objective_vals["final_longitude"])
        else:
            unfeasible_cross_range_array.append(objective_vals["cross_track_dist"])
            unfeasible_down_range_array.append(objective_vals["along_track_dist"])
            unfeasible_rsw_cross_range_array.append(objective_vals["rsw_cross_track_dist"])
            unfeasible_rsw_down_range_array.append(objective_vals["rsw_along_track_dist"])
            unfeasible_latitude_array.append(objective_vals["final_latitude"])
            unfeasible_longitude_array.append(objective_vals["final_longitude"])
            no_rejected += 1
        
    print(no_rejected)
    
    plt.figure()
    
    plt.title("lat/long")
    
    
    plt.scatter(feasible_longitude_array, feasible_latitude_array)
    plt.scatter(unfeasible_longitude_array, unfeasible_latitude_array, color="r")
    
    for i in range(len(good_trajectory_ids)):
        plt.annotate(f"{good_trajectory_ids[i]}", (feasible_longitude_array[i], feasible_latitude_array[i]))
        
    plt.ylabel("Latitude")
    plt.xlabel("Longitude")
        
    plt.figure()
    
    plt.title("RSW")
    
    plt.scatter(feasible_rsw_cross_range_array, feasible_rsw_down_range_array)
    plt.scatter(unfeasible_rsw_cross_range_array, unfeasible_rsw_down_range_array, color="r")
    
    plt.ylabel("downrange")
    plt.xlabel("crossrange")
    
    for i in range(len(good_trajectory_ids)):
        plt.annotate(f"{good_trajectory_ids[i]}", (feasible_rsw_cross_range_array[i], feasible_rsw_down_range_array[i]))

    plt.figure()
    plt.title("groundspeed")
    
    plt.scatter(feasible_cross_range_array, feasible_down_range_array)
    plt.scatter(unfeasible_cross_range_array, unfeasible_down_range_array, color="r")
    
    plt.ylabel("downrange")
    plt.xlabel("crossrange")
    
    for i in range(len(good_trajectory_ids)):
        plt.annotate(f"{good_trajectory_ids[i]}", (feasible_cross_range_array[i], feasible_down_range_array[i]))

    plt.show()


def create_dictionary_from_savefile(filepath):
    
    data = np.genfromtxt(filepath)
    dic = {}
    
    for row in data:
        dic[row[0]] = row[1:]

    return dic

def create_dic_drom_json(filepath):
    with open(filepath) as json_file:
        return json.load(json_file)
    

def save_dependent_variable_info(dictionary, filename):
    # Open the text file in write mode
    with open(filename, "w") as file:
        # Iterate over the dictionary items
        for key, value in dictionary.items():
            # Write each item to a new row in the text file
            file.write(f"{key}: {value}\n")   


def save_dict_to_json(data, filename):
    
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4, separators=(',', ': '))
        
        

def save_dynamics_simulator_to_files(output_path, dynamics_simulator):
    
    save2txt(dynamics_simulator.state_history, 'state_history.dat', output_path)
    save2txt(dynamics_simulator.dependent_variable_history, 'dependent_variable_history.dat', output_path)
        
        

if __name__ == "__main__":
    
    # find_footprint_area_for_multiple_trajectories(sim_data_dir + "/monte_carlo/shape_one_at_a_time/run_nose_radius/iter_0", 2**7)
    
    num_states_dic = create_dictionary_from_savefile(sim_data_dir + "/monte_carlo/shape_one_at_a_time/run_nose_radius/iter_0/trajectory_12/state_history.dat")
    dep_vars_dic = create_dictionary_from_savefile(sim_data_dir + "/monte_carlo/shape_one_at_a_time/run_nose_radius/iter_0/trajectory_12/dependent_variable_history.dat")
    decision_vars = create_dic_drom_json(sim_data_dir + "/monte_carlo/shape_one_at_a_time/run_nose_radius/iter_0/trajectory_12/decision_variables.dat")
    
    print(calculate_constraint_objective_functions(num_states_dic, dep_vars_dic, decision_vars))