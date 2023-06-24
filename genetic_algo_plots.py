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
        
    print(x_labels)
        
    plt.figure()
    
    plt.plot(x_labels, np.array(t_survive)/(24*60**2))
    
    plt.xticks(rotation = 90)
    plt.tight_layout()
    
    plt.figure()
    
    plt.plot(x_labels, np.array(period_t))
    
    plt.xticks(rotation = 90)
    plt.tight_layout()
    
    plt.show()
    
    
    pass



if __name__ == "__main__":
    
    plot_evolution_info(120, hf.sim_data_dir + f"/custom_genetic_algo/best_settings/version_1")
    
    # investigate_gen_algo_investigation()
    
    pass