import matplotlib.pyplot as plt
import numpy as np
import pickle
import helper_functions as hf
from gentic_algo import GA, get_fitness
import simulation as sim
import sys

def plot_evolution_info(generations, path_to_data):
    
    best_fitness = []
    t_survive = []
    period_t = []
    best_dv_r = []
    best_dv_s = []
    best_dv_w = []
    best_t_burn = []
    
    
    for gen in range(generations+1):
        with open(path_to_data + f"/gen_{gen}.pkl", 'rb') as f:
            generation = pickle.load(f)
            
        best_idx = np.argmin(generation.fitness_pool)  
        
        best_fitness.append(generation.fitness_pool[best_idx])          
        t_survive.append(generation.survival_pool[best_idx])          
        period_t.append(generation.constraint_pool[best_idx])          
        best_dv_r.append(generation.gene_pool[best_idx][0])         
        best_dv_s.append(generation.gene_pool[best_idx][1])         
        best_dv_w.append(generation.gene_pool[best_idx][2])         
        best_t_burn.append(generation.gene_pool[best_idx][3])        
        
    hf.plot_arrays(np.arange(0, generations+1), [np.array(best_fitness)/(24*60**2)], keep_in_memory=True, y_label="Fitness", x_label="Generations") 
    hf.plot_arrays(np.arange(0, generations+1), [np.array(t_survive)/(24*60**2)], keep_in_memory=True, y_label="Survival time", x_label="Generations") 
    hf.plot_arrays(np.arange(0, generations+1), [period_t], keep_in_memory=True, y_label="Delta T", x_label="Generations") 
    hf.plot_arrays(np.arange(0, generations+1), [best_dv_r], keep_in_memory=True, y_label="dv r", x_label="Generations") 
    hf.plot_arrays(np.arange(0, generations+1), [best_dv_s], keep_in_memory=True, y_label="dv s", x_label="Generations") 
    hf.plot_arrays(np.arange(0, generations+1), [best_dv_w], keep_in_memory=True, y_label="dv w", x_label="Generations") 
    hf.plot_arrays(np.arange(0, generations+1), [np.array(best_t_burn)/(24*60**2)], keep_in_memory=True, y_label="t burn", x_label="Generations") 
    
    plt.show()


def investigate_gen_algo_investigation(
    mutation_probability_array = np.linspace(0.01, 0.9, 20),
    breeding_parents_array = np.arange(2, 14, 2),
    no_parent_to_keep_array = np.arange(2, 10, 2),
    generations = 50,
    data_path = hf.external_sim_data_dir + "/custom_genetic_algo"
):
    
    x_labels = []
    best_fitness = []
    t_survive = []
    period_t = []
    best_dv_r = []
    best_dv_s = []
    best_dv_w = []
    best_t_burn = []
    
    for p in mutation_probability_array:
        with open(data_path + f"/mutation_probability_investigation/value={p}/gen_50.pkl", 'rb') as f:
            generation = pickle.load(f)
            
        best_idx = np.argmin(generation.fitness_pool)  
        best_fitness.append(generation.fitness_pool[best_idx])          
        t_survive.append(generation.survival_pool[best_idx])          
        period_t.append(generation.constraint_pool[best_idx])          
        best_dv_r.append(generation.gene_pool[best_idx][0])         
        best_dv_s.append(generation.gene_pool[best_idx][1])         
        best_dv_w.append(generation.gene_pool[best_idx][2])         
        best_t_burn.append(generation.gene_pool[best_idx][3])
        
        
        x_labels.append(str(f"p={p:.2e}"))   
        
    
    for b in breeding_parents_array:
        with open(data_path + f"/breeding_parents_investigation/value={b}/gen_50.pkl", 'rb') as f:
            generation = pickle.load(f)
            
        best_idx = np.argmin(generation.fitness_pool)  
        best_fitness.append(generation.fitness_pool[best_idx])          
        t_survive.append(generation.survival_pool[best_idx])          
        period_t.append(generation.constraint_pool[best_idx])          
        best_dv_r.append(generation.gene_pool[best_idx][0])         
        best_dv_s.append(generation.gene_pool[best_idx][1])         
        best_dv_w.append(generation.gene_pool[best_idx][2])         
        best_t_burn.append(generation.gene_pool[best_idx][3])     
        
        x_labels.append(str(f"b={b:.2e}"))  
    
    for no in no_parent_to_keep_array:
        with open(data_path + f"/no_parent_to_keep_investigation/value={no}/gen_50.pkl", 'rb') as f:
            generation = pickle.load(f)
            
        best_idx = np.argmin(generation.fitness_pool)  
        best_fitness.append(generation.fitness_pool[best_idx])          
        t_survive.append(generation.survival_pool[best_idx])          
        period_t.append(generation.constraint_pool[best_idx])          
        best_dv_r.append(generation.gene_pool[best_idx][0])         
        best_dv_s.append(generation.gene_pool[best_idx][1])         
        best_dv_w.append(generation.gene_pool[best_idx][2])         
        best_t_burn.append(generation.gene_pool[best_idx][3])     
        
        x_labels.append(str(f"no={no:.2e}"))  
        
    plt.figure()
    
    plt.plot(x_labels, np.array(t_survive)/(24*60**2))
    
    plt.xticks(rotation = 90)
    plt.tight_layout()
    
    plt.figure()
    
    plt.plot(x_labels, np.array(period_t))
    
    plt.xticks(rotation = 90)
    plt.tight_layout()
    
    plt.figure()
    
    plt.plot(x_labels, best_fitness)
    
    plt.xticks(rotation = 90)
    plt.tight_layout()

    plt.show()
    
    sys.exit()

    # plotting top 3 trajectories
    
    
    best_idx = np.argsort(best_fitness)  
    
    lats = []
    longs = []
    times = []
    
    
    for id in best_idx[:1]:
        decision_variable_dic = {'dv': np.array([best_dv_r[id], best_dv_s[id], best_dv_w[id]]), 't_impulse': best_t_burn[id]}        
        time, lat, long = sim.run_simulation(None, 6*31*24*60**2, decision_variable_dic=decision_variable_dic, 
                                             return_time_lat_long=True, run_for_mc=False)
        lats.append(lat)
        longs.append(long)
        times.append(np.array(time) / (24*60**2))
    
    
    hf.create_lat_long_circle_plot(lats, longs, times, colours=["tab:blue", "tab:green", "tab:orange"], keep_in_memory=True)
    plt.show()
    
    hf.create_animated_lat_long_circle_plot(
        lats, longs, times, 
        colours=["tab:blue", "tab:green", "tab:orange"], 
        path_to_save_animation=hf.report_dir + "/Figures/animated_pdf",
        filetype="pdf")
    
    
    
    pass


def plot_various_optimization_results(
    custom_data_path = hf.external_sim_data_dir + "/custom_genetic_algo/best_settings/version_1"    
):
    
    
    custom_gen_algo_info = hf.create_dic_drom_json(custom_data_path + "/evolution_info_dic.dat")
    
    best_fitness = []
    t_survive = []
    period_t = []
    best_dv_r = []
    best_dv_s = []
    best_dv_w = []
    best_t_burn = []
    
    for gen_id in range(custom_gen_algo_info["generations"]):
        with open(custom_data_path + f"/gen_{gen_id}.pkl", 'rb') as f:
            generation = pickle.load(f)
            
        best_idx = np.argmin(generation.fitness_pool)  
        best_fitness.append(generation.fitness_pool[best_idx])          
        t_survive.append(generation.survival_pool[best_idx]/(24*60**2))          
        period_t.append(generation.constraint_pool[best_idx])          
        best_dv_r.append(generation.gene_pool[best_idx][0])         
        best_dv_s.append(generation.gene_pool[best_idx][1])         
        best_dv_w.append(generation.gene_pool[best_idx][2])         
        best_t_burn.append(generation.gene_pool[best_idx][3])
    
    
    hf.plot_arrays(
        range(custom_gen_algo_info["generations"]),
        [best_fitness],
        x_label="Generations [-]",
        y_label="Best fitness in generation [s]",
        keep_in_memory=True)
    
    hf.plot_arrays(
        range(custom_gen_algo_info["generations"]),
        [t_survive],
        x_label="Generations [-]",
        y_label="Survival time for fittest in generation [days]",
        keep_in_memory=True)
    
    
    plt.show()
    
    
    pass


if __name__ == "__main__":
    
    # plot_evolution_info(80, hf.external_sim_data_dir + "/custom_genetic_algo/breeding_parents_investigation/value=12")
    
    # investigate_gen_algo_investigation()
    
    plot_various_optimization_results()
    
    pass