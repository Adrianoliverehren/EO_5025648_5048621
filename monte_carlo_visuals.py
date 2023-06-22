import helper_functions as hf
import numpy as np
import matplotlib.pyplot as plt










def create_genera_info_plots(
    data_path = hf.external_sim_data_dir + "/DesignSpace/monte_carlo",
    no_sims = 2**12
):


    # for iter in range(no_sims):
    #     iter_path = data_path + f"/iter_{iter}"
        
    
    objective_constraints = np.genfromtxt(data_path + "/objectives_constraints.dat").T
    
    parameter_values = hf.create_dic_drom_json(data_path + "/parameter_values.dat")
    

    dv_mag = np.zeros(no_sims)
    r_dir = np.zeros(no_sims)
    s_dir = np.zeros(no_sims)
    w_dir = np.zeros(no_sims)
    t_impulse = np.zeros(no_sims)
    
    unfeasible_ids = []
    
    i = 0
    for key, value in parameter_values.items():
        
        dv_mag[int(key)] = value["dv_mag"]
        r_dir[int(key)] = value["dv_unit_vect"][0]
        s_dir[int(key)] = value["dv_unit_vect"][1]
        w_dir[int(key)] = value["dv_unit_vect"][2]
        t_impulse[int(key)] = value["t_impulse"]
        if objective_constraints[2][i] > 0:
            unfeasible_ids.append(i)
        i += 1
        
    
    hf.plot_arrays(dv_mag, [objective_constraints[1] / (24*60**2)], linewiths=[0]*len(dv_mag), markings=True, keep_in_memory=True,
                    y_label="Survival time [days]", x_label="dv_mag")
    
    hf.plot_arrays(r_dir, [objective_constraints[1] / (24*60**2)], linewiths=[0]*len(dv_mag), markings=True, keep_in_memory=True,
                    y_label="Survival time [days]", x_label="r_dir")
    
    hf.plot_arrays(s_dir, [objective_constraints[1] / (24*60**2)], linewiths=[0]*len(dv_mag), markings=True, keep_in_memory=True,
                    y_label="Survival time [days]", x_label="s_dir")
    
    hf.plot_arrays(w_dir, [objective_constraints[1] / (24*60**2)], linewiths=[0]*len(dv_mag), markings=True, keep_in_memory=True,
                    y_label="Survival time [days]", x_label="w_dir")
    
    hf.plot_arrays(r_dir * dv_mag, [objective_constraints[1] / (24*60**2)], linewiths=[0]*len(dv_mag), markings=True, keep_in_memory=True,
                    y_label="Survival time [days]", x_label="dv_r_dir")
    
    hf.plot_arrays(s_dir * dv_mag, [objective_constraints[1] / (24*60**2)], linewiths=[0]*len(dv_mag), markings=True, keep_in_memory=True,
                    y_label="Survival time [days]", x_label="dv_s_dir")
    
    hf.plot_arrays(w_dir * dv_mag, [objective_constraints[1] / (24*60**2)], linewiths=[0]*len(dv_mag), markings=True, keep_in_memory=True,
                    y_label="Survival time [days]", x_label="dv_w_dir")
    
    hf.plot_arrays(t_impulse / (24*60**2), [objective_constraints[1] / (24*60**2)], linewiths=[0]*len(dv_mag), markings=True, keep_in_memory=True,
                    y_label="Survival time [days]", x_label="t_impulse")
    
    
    
    hf.plot_heatmap_scatter(r_dir * dv_mag, t_impulse / (24*60**2), [objective_constraints[1] / (24*60**2)], keep_in_memory=True, x_label="R dV",
                            y_label="Time of inpulse [days]", z_label="Survival time [days]", filter_ids=unfeasible_ids)
    hf.plot_heatmap_scatter(s_dir * dv_mag, t_impulse / (24*60**2), [objective_constraints[1] / (24*60**2)], keep_in_memory=True, x_label="S dV",
                            y_label="Time of inpulse [days]", z_label="Survival time [days]", filter_ids=unfeasible_ids)
    hf.plot_heatmap_scatter(w_dir * dv_mag, t_impulse / (24*60**2), [objective_constraints[1] / (24*60**2)], keep_in_memory=True, x_label="W dV",
                            y_label="Time of inpulse [days]", z_label="Survival time [days]", filter_ids=unfeasible_ids)
    
    
    
    plt.show()
    
    pass



if __name__ == "__main__":
    
    create_genera_info_plots()