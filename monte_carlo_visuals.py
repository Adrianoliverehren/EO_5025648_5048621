import helper_functions as hf
import numpy as np
import matplotlib.pyplot as plt




def create_one_at_a_time_plots(
    data_path = hf.external_sim_data_dir + "/DesignSpace/monte_carlo_one_at_a_time",
    no_sims = 2**7
):
    
    objective_constraints = np.genfromtxt(data_path + "/objectives_constraints.dat").T
    
    parameter_values = hf.create_dic_drom_json(data_path + "/parameter_values.dat")
    
    dv_r_dir = np.zeros(no_sims)
    dv_s_dir = np.zeros(no_sims)
    dv_w_dir = np.zeros(no_sims)
    t_impulse = np.zeros(no_sims)
    t_survive = [np.zeros(no_sims), np.zeros(no_sims), np.zeros(no_sims), np.zeros(no_sims)]
    unfeasible_ids = [[],[],[],[]]
    ids_to_delete = [[],[],[],[]]
    
    for iter in range(no_sims):
            
        dv_r_dir[iter] = parameter_values[str(iter+0*no_sims)]["dv"][0]
        dv_s_dir[iter] = parameter_values[str(iter+1*no_sims)]["dv"][1]
        dv_w_dir[iter] = parameter_values[str(iter+2*no_sims)]["dv"][2]
        t_impulse[iter] = parameter_values[str(iter+3*no_sims)]["t_impulse"] / (24*60**2) 
        
        t_survive[0][iter] = objective_constraints[1][iter+0*no_sims] / (24*60**2)
        t_survive[1][iter] = objective_constraints[1][iter+1*no_sims] / (24*60**2)
        t_survive[2][iter] = objective_constraints[1][iter+2*no_sims] / (24*60**2)
        t_survive[3][iter] = objective_constraints[1][iter+3*no_sims] / (24*60**2)
        
        if objective_constraints[2][iter+0*no_sims] > 0:
            unfeasible_ids[0].append(iter)
        if objective_constraints[2][iter+1*no_sims] > 0:
            unfeasible_ids[1].append(iter)
        if objective_constraints[2][iter+2*no_sims] > 0:
            unfeasible_ids[2].append(iter)
        if objective_constraints[2][iter+3*no_sims] > 0 and objective_constraints[2][iter+3*no_sims] != np.inf:
            unfeasible_ids[3].append(iter)
        if objective_constraints[2][iter+3*no_sims] == np.inf:
            ids_to_delete[3].append(iter)
        
    
    dv_r_dir_feasible = np.delete(dv_r_dir, unfeasible_ids[0])
    dv_r_dir_unfeasible = np.take(dv_r_dir, unfeasible_ids[0])
    
    dv_s_dir_feasible = np.delete(dv_s_dir, unfeasible_ids[1])
    dv_s_dir_unfeasible = np.take(dv_s_dir, unfeasible_ids[1])
    
    dv_w_dir_feasible = np.delete(dv_w_dir, unfeasible_ids[2])
    dv_w_dir_unfeasible = np.take(dv_w_dir, unfeasible_ids[2])
    
    t_impulse[ids_to_delete[3]] = np.nan
    t_impulse_feasible = np.delete(t_impulse, unfeasible_ids[3])
    t_impulse_unfeasible = np.take(t_impulse, unfeasible_ids[3])
    
    t_survive_feasible = [[]]*4
    t_survive_unfeasible = [[]]*4
    
    t_survive_feasible[0] = np.delete(t_survive[0], unfeasible_ids[0])
    t_survive_unfeasible[0] = np.take(t_survive[0], unfeasible_ids[0])
    
    t_survive_feasible[1] = np.delete(t_survive[1], unfeasible_ids[1])
    t_survive_unfeasible[1] = np.take(t_survive[1], unfeasible_ids[1])
    
    t_survive_feasible[2] = np.delete(t_survive[2], unfeasible_ids[2])
    t_survive_unfeasible[2] = np.take(t_survive[2], unfeasible_ids[2])
    
    t_survive[3][ids_to_delete[3]] = np.nan
    t_survive_feasible[3] = np.delete(t_survive[3], unfeasible_ids[3])
    t_survive_unfeasible[3] = np.take(t_survive[3], unfeasible_ids[3])
            
    
    hf.plot_arrays([dv_r_dir_unfeasible, dv_r_dir_feasible], [t_survive_unfeasible[0], t_survive_feasible[0]], linewiths=[0]*len(dv_r_dir), markings=True, keep_in_memory=False,
                    y_label="Survival time [days]", x_label=r"$\Delta V$ in radial direction [m/s]", colors=["gray", "tab:blue"], alphas=[0.2, 1], legend=["unfeasible", "feasible"],
                    path_to_save=hf.report_dir + "/Figures/Ch2/monte_carlo/one_at_a_time/dv_r_dir_vs_survive_time.pdf")
    
    hf.plot_arrays([dv_s_dir_unfeasible, dv_s_dir_feasible], [t_survive_unfeasible[1], t_survive_feasible[1]], linewiths=[0]*len(dv_r_dir), markings=True, keep_in_memory=False,
                    y_label="Survival time [days]", x_label=r"$\Delta V$ in along track direction [m/s]", colors=["gray", "tab:blue"], alphas=[0.2, 1], legend=["unfeasible", "feasible"],
                    path_to_save=hf.report_dir + "/Figures/Ch2/monte_carlo/one_at_a_time/dv_s_dir_vs_survive_time.pdf")
    
    hf.plot_arrays([dv_w_dir_unfeasible, dv_w_dir_feasible], [t_survive_unfeasible[2], t_survive_feasible[2]], linewiths=[0]*len(dv_r_dir), markings=True, keep_in_memory=False,
                    y_label="Survival time [days]", x_label=r"$\Delta V$ in cross track direction [m/s]", colors=["gray", "tab:blue"], alphas=[0.2, 1], legend=["unfeasible", "feasible"],
                    path_to_save=hf.report_dir + "/Figures/Ch2/monte_carlo/one_at_a_time/dv_w_dir_vs_survive_time.pdf")
    
    hf.plot_arrays([t_impulse_unfeasible, t_impulse_feasible], [t_survive_unfeasible[3], t_survive_feasible[3]], linewiths=[0]*len(dv_r_dir), markings=True, keep_in_memory=False,
                    y_label="Survival time [days]", x_label="Time of impulse [days]", colors=["gray", "tab:blue"], alphas=[0.2, 1], legend=["unfeasible", "feasible"],
                    path_to_save=hf.report_dir + "/Figures/Ch2/monte_carlo/one_at_a_time/t_impulse_vs_survive_time.pdf")

def create_general_info_plots(
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
    
    hf.plot_arrays([dv_r_dir_unfeasible, dv_r_dir_feasible], [t_survive_unfeasible, t_survive_feasible], linewiths=[0]*len(dv_r_dir), markings=True, keep_in_memory=False,
                    y_label="Survival time [days]", x_label=r"$\Delta V$ in radial direction [m/s]", colors=["gray", "tab:blue"], alphas=[0.2, 1],
                    path_to_save=hf.report_dir + "/Figures/Ch2/monte_carlo/all_at_once/R_vs_survival_time.pdf")
    
    hf.plot_arrays([dv_s_dir_unfeasible, dv_s_dir_feasible], [t_survive_unfeasible, t_survive_feasible], linewiths=[0]*len(dv_r_dir), markings=True, keep_in_memory=False,
                    y_label="Survival time [days]", x_label=r"$\Delta V$ in along track direction [m/s]", colors=["gray", "tab:blue"], alphas=[0.2, 1],
                    path_to_save=hf.report_dir + "/Figures/Ch2/monte_carlo/all_at_once/S_vs_survival_time.pdf")
    
    hf.plot_arrays([dv_w_dir_unfeasible, dv_w_dir_feasible], [t_survive_unfeasible, t_survive_feasible], linewiths=[0]*len(dv_r_dir), markings=True, keep_in_memory=False,
                    y_label="Survival time [days]", x_label=r"$\Delta V$ in cross track direction [m/s]", colors=["gray", "tab:blue"], alphas=[0.2, 1],
                    path_to_save=hf.report_dir + "/Figures/Ch2/monte_carlo/all_at_once/W_vs_survival_time.pdf")
    
    hf.plot_arrays([t_impulse_unfeasible, t_impulse_feasible], [t_survive_unfeasible, t_survive_feasible], linewiths=[0]*len(dv_r_dir), markings=True, keep_in_memory=False,
                    y_label="Survival time [days]", x_label="Time of impulse [days]", colors=["gray", "tab:blue"], alphas=[0.2, 1],
                    path_to_save=hf.report_dir + "/Figures/Ch2/monte_carlo/all_at_once/T_impulse_vs_survival_time.pdf")
    
    
    hf.plot_heatmap_scatter(dv_r_dir, t_impulse / (24*60**2), [objective_constraints[1] / (24*60**2)], keep_in_memory=False, x_label=r"$\Delta V$ in radial direction [m/s]",
                            y_label="Time of impulse [days]", z_label="Survival time [days]", filter_ids=unfeasible_ids,
                            path_to_save=hf.report_dir + "/Figures/Ch2/monte_carlo/all_at_once/R_and_t_impulse_vs_survival_time.pdf", plot_size=[4,4.5], marker_size=20)
    hf.plot_heatmap_scatter(dv_s_dir, t_impulse / (24*60**2), [objective_constraints[1] / (24*60**2)], keep_in_memory=False, x_label=r"$\Delta V$ in along track direction [m/s]",
                            y_label="Time of impulse [days]", z_label="Survival time [days]", filter_ids=unfeasible_ids,
                            path_to_save=hf.report_dir + "/Figures/Ch2/monte_carlo/all_at_once/S_and_t_impulse_vs_survival_time.pdf", plot_size=[4,4.5], marker_size=20)
    hf.plot_heatmap_scatter(dv_w_dir, t_impulse / (24*60**2), [objective_constraints[1] / (24*60**2)], keep_in_memory=False, x_label=r"$\Delta V$ in cross track direction [m/s]",
                            y_label="Time of impulse [days]", z_label="Survival time [days]", filter_ids=unfeasible_ids,
                            path_to_save=hf.report_dir + "/Figures/Ch2/monte_carlo/all_at_once/W_and_t_impulse_vs_survival_time.pdf", plot_size=[4,4.5], marker_size=20)
    
    
    
    pass



if __name__ == "__main__":
    
    create_one_at_a_time_plots()
    
    create_general_info_plots()