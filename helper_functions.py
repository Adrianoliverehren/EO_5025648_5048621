import pdb
import warnings
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
import tudatpy.kernel.interface.spice as spice
from tudatpy.kernel.astro.element_conversion import cartesian_to_keplerian
from matplotlib import colors
import matplotlib.animation as animation
from PIL import Image
import glob
from natsort import natsorted

def remove_folder_from_path(path : str, no_to_remove=1):
    path = path.replace("\\", "/")
    if no_to_remove >= path.count('/'):
        path = '/'
    else:
        for _ in range(no_to_remove):
            path = os.path.dirname(path)    
    return path

root_dir = os.path.dirname(__file__)
report_dir = remove_folder_from_path(os.path.dirname(__file__), 1) + "/overleaf"
sim_data_dir = os.path.dirname(__file__) + "/SimulationData"
plots_dir = os.path.dirname(__file__) + "/SimulationData/plots/"
external_sim_data_dir = remove_folder_from_path(os.path.dirname(__file__), 1) + "/SimulationData"

def plot_arrays(
    x_arrays, y_arrays, path_to_save=False, title=None, x_label=None, y_label=None, scale_factor=None,
    legend=None, grid=True, x_log=False, linestyles=None, linewiths=None, plot_size=[4,4], colors=None, x_lim=None, legend_pos=None,
    force_xticks=False, force_sci_notation=False, custom_legend_entries=None, custom_markings=None, markers=None, marker_colors=None,
    markerfacecolors=None, marker_sizes=None, keep_in_memory=False, y_log=False, markings=None, additional_save_path=None, 
    alphas=None, y_simlog=False, **kwargs):
    
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
        if alphas:
            if len(alphas) == 1:
                alphas = alphas * len(y_arrays)
            plt.setp(line_plot, alpha=alphas[i])
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
        
    if y_simlog:
        plt.yscale("symlog")
        
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
    normalize_colourbar=None, z_label=None, filter_ids=None, marker_size=30, **kwargs):
    
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
        
        # plt.scatter(x_unwanted, y_unwanted, c="gray", marker=".")
        
        x_array = np.delete(x_array, filter_ids)
        y_array = np.delete(y_array, filter_ids)
        z_array = np.delete(z_array, filter_ids)
        

    if normalize_colourbar=="start_at_zero":
        
        ids_below_zero = np.argwhere(z_array < 0).T[0]
        ids_above_zero = np.argwhere(z_array >= 0).T[0]
        
        
        x_below_zero = np.take(x_array, ids_below_zero)
        y_below_zero = np.take(y_array, ids_below_zero)
        z_below_zero = np.take(z_array, ids_below_zero)
        
        plt.scatter(x_below_zero, y_below_zero, c="gray", marker=".", s=marker_size)
        x_array = np.take(x_array, ids_above_zero)
        y_array = np.take(y_array, ids_above_zero)
        z_array = np.take(z_array, ids_above_zero)
    
        plt.scatter(x_array, y_array, c=z_array, cmap=reversed_map, s=marker_size)
    
    if normalize_colourbar=="log":
        
        
        ids_non_zero = np.argwhere(z_array > 1e-15).T[0]
        x_array = np.take(x_array, ids_non_zero)
        y_array = np.take(y_array, ids_non_zero)
        z_array = np.take(z_array, ids_non_zero)
        
        # sys.exit()
        norm=colors.LogNorm(vmin=np.min(z_array), 
        vmax=np.max(z_array))
        
        plt.scatter(x_array, y_array, c=z_array, cmap=reversed_map, norm=norm, s=marker_size)
    
    else:
        
        plt.scatter(x_array, y_array, c=z_array, cmap=reversed_map, s=marker_size)
    

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


def create_lat_long_circle_plot(lat_arrays, long_arrays, time_array, colours, animate="no", path_to_save=None, keep_in_memory=False):

    plt.figure()

    ax = plt.axes(projection='3d')
    
    c_array = np.linspace(0, 2*np.pi, 200)
    
    x_lim = np.cos(c_array) * constraint() * np.rad2deg(1)
    y_lim = np.sin(c_array) * constraint() * np.rad2deg(1)

    # Data for a three-dimensional line
    
    
    big_time_list = []
    
    for t in time_array:
        big_time_list += list(t)    
        
    long_list = []
    
    for lat, long in zip(lat_arrays, long_arrays):
        long_list += list(lat)
        long_list += list(long)
    
    maxmax = max(np.abs(long_list))*1.1* np.rad2deg(1)
    minmin = -maxmax
        
    ax.plot3D(x_lim, y_lim, [time_array[0][0]]*len(x_lim), 'red', alpha=0.4)
    
    ax.view_init(elev=20., azim=-135.)
       
    
    max_time = max(big_time_list)
    min_time = min(big_time_list)
    
    # lim lat /lon lines along z axis
    ax.plot3D([constraint() * np.rad2deg(1)]*2, [maxmax]*2, [min_time, max_time], 'red', alpha=0.4)
    ax.plot3D([-constraint() * np.rad2deg(1)]*2, [maxmax]*2, [min_time, max_time], 'red', alpha=0.4)
    
    ax.plot3D([maxmax]*2, [constraint() * np.rad2deg(1)]*2, [min_time, max_time], 'red', alpha=0.4)
    ax.plot3D([maxmax]*2, [-constraint() * np.rad2deg(1)]*2, [min_time, max_time], 'red', alpha=0.4)
    
    def get_animation(frame):
        for lat, long, t, c in zip(lat_arrays, long_arrays, time_array, colours):
            id_to_slice = int(frame/100 * len(lat))   
            lat, long = lat*np.rad2deg(1), long*np.rad2deg(1)
        
            ax.plot3D(lat[:id_to_slice], ([maxmax]*len(lat))[:id_to_slice], t[:id_to_slice], c, alpha=0.4)
            ax.plot3D(([maxmax]*len(lat))[:id_to_slice], long[:id_to_slice], t[:id_to_slice], c, alpha=0.4)
        for lat, long, t, c in zip(lat_arrays, long_arrays, time_array, colours):
            
            
            id_to_slice = int(frame/100 * len(lat))   
            lat, long = lat*np.rad2deg(1), long*np.rad2deg(1)
            
            ax.plot3D(lat[:id_to_slice], long[:id_to_slice], t[:id_to_slice], c)
    
    if animate == "no":
        get_animation(100)
    else:
        get_animation(animate)
        
    
    ax.set_xlabel("Latitude [deg]")
    ax.set_ylabel("Longitude [deg]")
    ax.set_zlabel("Time [days]")
    
    ax.set_xlim(xmin=minmin, xmax=maxmax)
    ax.set_ylim(ymin=minmin, ymax=maxmax)
    
    plt.tight_layout()
        
    if path_to_save:        
        path = pathlib.Path(path_to_save)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_to_save)

    if keep_in_memory == False:
        plt.close()
    
    pass


def make_gif_from_pngs(image_directory):
    

    # List all image files in the directory
    image_files = glob.glob(image_directory + "*.png")  # Change the extension according to your image format

    # Sort the image files based on their names (if necessary)
    image_files = natsorted(image_files)

    # Create an empty list to store the images
    images = []
    
    print(image_files)

    # Open and append each image to the list
    for filename in image_files:
        img = Image.open(filename)
        images.append(img)
        

    # Save the images as a GIF
    output_file = "output.gif"

    # Use the save method of the first image, and pass the other images as frames
    images[0].save(image_directory + output_file, save_all=True, append_images=images[1:], duration=200, loop=0)


def create_animated_lat_long_circle_plot(lat_arrays, long_arrays, time_array, colours, path_to_save_animation, filetype="png"):
    
    i = 0
    for ix in np.linspace(0, 100, 200):
        
        path_to_save = path_to_save_animation + f"/frame_{i}.{filetype}"
        create_lat_long_circle_plot(lat_arrays, long_arrays, time_array, colours, animate=ix, path_to_save=path_to_save, keep_in_memory=False)
        
        i+=1
    
    pass

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

def make_dic_safe_for_json(dictionary):
    safe_dic = {}
    # Open the text file in write mode
    for key, value in dictionary.items():
        # Write each item to a new row in the text file
        if type(value) == np.ndarray:
            value = value.tolist()
        safe_dic[str(key)] = value
    return safe_dic
    
def save_dict_to_json(data, filename):
    path = pathlib.Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4, separators=(',', ': '))
        
def save_dynamics_simulator_to_files(output_path, stacked_state_history, stacked_dep_vars_history):
        save2txt(stacked_state_history, 'state_history.dat', output_path)
        save2txt(stacked_dep_vars_history, 'dependent_variable_history.dat', output_path)

def constraint():
    return 7.589381 * (10**(-4)) # 32 / semi-major axis

def make_ALL_folders_for_path(filepath):
    path = pathlib.Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)   
    
def calculate_obj(dependent_var_history, sim_idx="n/a"):
    for t in dependent_var_history.keys():
        current_dep_vars = dependent_var_history[t]
        angle_radius_thing = np.sqrt(current_dep_vars[5]**2 + current_dep_vars[6]**2)
        if angle_radius_thing > constraint():
            return t

    # If condition was never violated
    warnings.warn(f'Simulation #{sim_idx} never exceeded position constraint, returned final time')
    return list(dependent_var_history.keys())[-1]

def period_change(state_history, t_impulse, dependent_var_history):
    t_arr = np.array(list(state_history.keys()))
    state_history_arr = np.array(list(state_history.values()))

    # Determine initial period
    mu = spice.get_body_gravitational_parameter('Earth')
    kepler_state_initial = cartesian_to_keplerian(state_history_arr[0], mu)
    T_initial = 2 * np.pi * (kepler_state_initial[0]**3 / mu)**0.5

    if t_impulse > calculate_obj(dependent_var_history):
        return np.inf
    # Determine T prior to impulse
    prior_index = np.max(np.argwhere(t_arr < t_impulse))
    kepler_state_prior = cartesian_to_keplerian(state_history_arr[prior_index], mu)
    T_prior = 2 * np.pi * (kepler_state_prior[0]**3 / mu)**0.5

    # Determine T after impulse
    after_index = np.min(np.argwhere(t_arr > t_impulse))

    kepler_state_after = cartesian_to_keplerian(state_history_arr[after_index], mu)
    T_after = 2 * np.pi * (kepler_state_after[0]**3 / mu)**0.5

    """
    dT_arr = np.zeros(len(state_history_arr[:, 0]))
    for idx, state in enumerate(state_history_arr):
        kepler_state_current = cartesian_to_keplerian(state, mu)
        T_current = (2 * np.pi * kepler_state_current[0]**3 / mu)**0.5
        dT_arr[idx] = T_current - T_initial
    """
    return abs(T_after - T_initial) - abs(T_prior - T_initial)        # max(abs(dT_arr))


if __name__ == "__main__":
    
    
    make_gif_from_pngs(external_sim_data_dir + "/custom_genetic_algo/animation/")
    
    # print(np.rad2deg(constraint()))