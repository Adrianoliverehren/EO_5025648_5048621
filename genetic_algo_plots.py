import matplotlib.pyplot as plt
import numpy as np
import pickle
import helper_functions as hf
from gentic_algo import GA, get_fitness



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





if __name__ == "__main__":
    
    plot_evolution_info(80, hf.sim_data_dir + "/custom_genetic_algo/test_run2")
    
    pass