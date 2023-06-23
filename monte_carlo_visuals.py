import helper_functions as hf
import numpy as np
import matplotlib.pyplot as plt










def create_genera_info_plots(
    data_path = hf.external_sim_data_dir + "/DesignSpace/monte_carlo",
    no_sims = 2**14
):


    # for iter in range(no_sims):
    #     iter_path = data_path + f"/iter_{iter}"
        
    
    objective_constraints = np.genfromtxt(data_path + "/objectives_constraints.dat").T
    
    parameter_values = hf.create_dic_drom_json(data_path + "/parameter_values.dat")
    
    dv_r_dir = np.zeros(no_sims)
    dv_s_dir = np.zeros(no_sims)
    dv_w_dir = np.zeros(no_sims)
    t_impulse = np.zeros(no_sims)
    
    unfeasible_ids = []
    
    i = 0
    for key, value in parameter_values.items():
        
        dv_r_dir[int(key)] = value["dv"][0]
        dv_s_dir[int(key)] = value["dv"][1]
        dv_w_dir[int(key)] = value["dv"][2]
        t_impulse[int(key)] = value["t_impulse"]
        if objective_constraints[2][i] > 0:
            unfeasible_ids.append(i)
        i += 1
        
    t_survive = objective_constraints[1] / (24*60**2)
    
    t_survive_feasible = np.delete(t_survive, unfeasible_ids)
    t_survive_unfeasible = np.take(t_survive, unfeasible_ids)
    
    dv_r_dir_feasible = np.delete(dv_r_dir, unfeasible_ids)
    dv_r_dir_unfeasible = np.take(dv_r_dir, unfeasible_ids)
    
    dv_s_dir_feasible = np.delete(dv_s_dir, unfeasible_ids)
    dv_s_dir_unfeasible = np.take(dv_s_dir, unfeasible_ids)
    
    dv_w_dir_feasible = np.delete(dv_w_dir, unfeasible_ids)
    dv_w_dir_unfeasible = np.take(dv_w_dir, unfeasible_ids)
    
    t_impulse_feasible = np.delete(t_impulse, unfeasible_ids)
    t_impulse_unfeasible = np.take(t_impulse, unfeasible_ids)
    
    # x_unwanted = np.take(x_array, filter_ids)
    # x_array = np.delete(x_array, filter_ids)
    
    hf.plot_arrays([dv_r_dir_unfeasible, dv_r_dir_feasible], [t_survive_unfeasible, t_survive_feasible], linewiths=[0]*len(dv_r_dir), markings=True, keep_in_memory=True,
                    y_label="Survival time [days]", x_label="dv_r_dir", colors=["gray", "tab:blue"], alphas=[0.2, 1])
    
    hf.plot_arrays([dv_s_dir_unfeasible, dv_s_dir_feasible], [t_survive_unfeasible, t_survive_feasible], linewiths=[0]*len(dv_r_dir), markings=True, keep_in_memory=True,
                    y_label="Survival time [days]", x_label="dv_s_dir", colors=["gray", "tab:blue"], alphas=[0.2, 1])
    
    hf.plot_arrays([dv_w_dir_unfeasible, dv_w_dir_feasible], [t_survive_unfeasible, t_survive_feasible], linewiths=[0]*len(dv_r_dir), markings=True, keep_in_memory=True,
                    y_label="Survival time [days]", x_label="dv_w_dir", colors=["gray", "tab:blue"], alphas=[0.2, 1])
    
    hf.plot_arrays([t_impulse_unfeasible, t_impulse_feasible], [t_survive_unfeasible, t_survive_feasible], linewiths=[0]*len(dv_r_dir), markings=True, keep_in_memory=True,
                    y_label="Survival time [days]", x_label="t_impulse", colors=["gray", "tab:blue"], alphas=[0.2, 1])
    
    
    
    
    hf.plot_heatmap_scatter(dv_r_dir, t_impulse / (24*60**2), [objective_constraints[1] / (24*60**2)], keep_in_memory=True, x_label="R dV",
                            y_label="Time of inpulse [days]", z_label="Survival time [days]", filter_ids=unfeasible_ids)
    hf.plot_heatmap_scatter(dv_s_dir, t_impulse / (24*60**2), [objective_constraints[1] / (24*60**2)], keep_in_memory=True, x_label="S dV",
                            y_label="Time of inpulse [days]", z_label="Survival time [days]", filter_ids=unfeasible_ids)
    hf.plot_heatmap_scatter(dv_w_dir, t_impulse / (24*60**2), [objective_constraints[1] / (24*60**2)], keep_in_memory=True, x_label="W dV",
                            y_label="Time of inpulse [days]", z_label="Survival time [days]", filter_ids=unfeasible_ids)
    
    
    
    plt.show()
    
    pass



if __name__ == "__main__":
    
    create_genera_info_plots()