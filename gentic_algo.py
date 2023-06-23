import numpy as np
import itertools
import math
import simulation as sim
import multiprocessing as mp
import helper_functions as hf
import pickle

def get_fitness(decision_var_arr):
    """Wraps simulation function to allow for input and output format necessary for scipy optimization algorithms"""

    decision_var_dict = {'dv': np.array([decision_var_arr[0], decision_var_arr[1], decision_var_arr[2]]),
                         't_impulse': decision_var_arr[3]}

    _, [unpenalized_objective, constraint] = sim.run_simulation(False, 6 * 31 * 24 * 60**2,
                                                                decision_variable_dic=decision_var_dict)

    if constraint > 0:
        penalty = constraint**2 * 10**6
    else:
        penalty = 0
    fitness = -unpenalized_objective + penalty
    return fitness, unpenalized_objective, constraint


class GA:
    def __init__(
        self,
        fitness_function,
        genes_array_boundaries,
        seed="random",
        ) -> None:
                
        self.fitness_function = fitness_function
        self.genes_array_boundaries = genes_array_boundaries
        if seed != "random":
            self.np_rand = np.random.RandomState(seed)
        else:
            self.np_rand = np.random.RandomState()
            
    
    def select_best_mates(self, no_mates):    
            
        sorted_ids = np.argsort(self.fitness_pool)
                
        self.mating_genes = []
        self.mating_fitnie = []
        
        for i in range(no_mates):
            self.mating_genes.append(self.gene_pool[sorted_ids[i]])
            self.mating_fitnie.append(self.fitness_pool[sorted_ids[i]])
        
    def evaluate_genes(self, genes, mp_nodes):
        fitness = []
        survival_time = []
        constraint = []
        
        input_lst = []
        
        for gene in genes:
            input_lst.append([gene])
            
        async_results=[]
        
        if mp_nodes:    
            pool = mp.Pool(mp_nodes, maxtasksperchild=40)
            for inp in input_lst:
                async_results.append(pool.apply_async(self.fitness_function, args=inp))
            pool.close()
            pool.join() 
            for async_res in async_results:
                fitness.append(async_res.get()[0])
                survival_time.append(async_res.get()[1])
                constraint.append(async_res.get()[2])
        else:
            for i, inp in enumerate(input_lst):
                out = self.fitness_function(*inp)
                fitness.append(out[0])  
                survival_time.append(out[1])  
                constraint.append(out[2])  
                        
        return fitness, survival_time, constraint
    
    def evolve(self, no_mates, no_children, mutation_probability, no_best_parents_to_keep, mp_nodes):
        
        self.select_best_mates(no_mates)
        
        # breed        
        fuck_list = list(itertools.combinations_with_replacement(np.arange(0, 5), 2))
        fuck_list = [c for c in fuck_list if c[0] != c[1]]
        frac = no_children / len(fuck_list)
        fuck_list = fuck_list * math.ceil(frac)
        
        children_genes = []
        
        for i in range(no_children):
        
            couple = fuck_list[i]
            
            gene_selection = np.random.randint(2, size=len(self.genes_array_boundaries))
            new_child = []
            
            for j, parent_id in enumerate(gene_selection):
                if np.random.random() < mutation_probability: 
                    new_gene = np.random.uniform(self.genes_array_boundaries[j][0], self.genes_array_boundaries[j][1])
                else:
                    new_gene = self.mating_genes[couple[parent_id]][j]
                    
                new_child.append(new_gene)
                
                pass
                
            children_genes.append(new_child)
        
        children_fitness, children_survival_time, children_constraint = self.evaluate_genes(children_genes, mp_nodes)
        
        children_sorted_ids = np.argsort(children_fitness)
        parents_sorted_ids = np.argsort(self.fitness_pool)
        
        if no_best_parents_to_keep > 0:
            for i in range(no_best_parents_to_keep):
                #check if worst child is less fit than best parent if so replace the child
                if children_fitness[children_sorted_ids[no_children - i - 1]] > self.fitness_pool[parents_sorted_ids[i]]:
                    children_fitness[children_sorted_ids[no_children - i - 1]] = self.fitness_pool[parents_sorted_ids[i]]
                    children_genes[children_sorted_ids[no_children - i - 1]] = self.gene_pool[parents_sorted_ids[i]]
                    children_survival_time[children_sorted_ids[no_children - i - 1]] = self.survival_pool[parents_sorted_ids[i]]
                    children_constraint[children_sorted_ids[no_children - i - 1]] = self.constraint_pool[parents_sorted_ids[i]]
                    
        self.fitness_pool = children_fitness
        self.gene_pool = children_genes
        self.survival_pool = children_survival_time
        self.constraint_pool = children_constraint
        
    
    def evolve_population(
        self, 
        individuals, 
        generations, 
        no_mates, 
        mutation_probability,
        no_best_parents_to_keep,
        mp_nodes=None,
        path_to_save_data=None):
        
        self.gene_pool = []
        
        
        
        for i in range(individuals):
            new_individual = np.zeros(len(self.genes_array_boundaries))
            for j in range(len(self.genes_array_boundaries)):
                new_individual[j] = np.random.uniform(self.genes_array_boundaries[j][0], self.genes_array_boundaries[j][1])
            
            self.gene_pool.append(new_individual)
        
        self.fitness_pool, self.survival_pool, self.constraint_pool = self.evaluate_genes(self.gene_pool, mp_nodes)
            
        pickle_file = path_to_save_data + f"/gen_0.pkl"
        
        hf.make_ALL_folders_for_path(pickle_file)
    
        with open(pickle_file, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
            
        evolution_info_dic = {
            "individuals": individuals,
            "generations": generations,
            "no_mates": no_mates,
            "mutation_probability": mutation_probability,
            "no_best_parents_to_keep": no_best_parents_to_keep,
            "mp_nodes": mp_nodes,
        }
            
        hf.save_dict_to_json(evolution_info_dic, path_to_save_data + f"/evolution_info_dic.dat")
        
        for gen in range(generations):
            self.evolve(no_mates, individuals, mutation_probability, no_best_parents_to_keep, mp_nodes)
            
            pickle_file = path_to_save_data + f"/gen_{gen+1}.pkl"
            
            hf.make_ALL_folders_for_path(pickle_file)
        
            with open(pickle_file, 'wb') as outp:
                pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
            
            

        pass
        

        
if __name__ == "__main__":
    
    bounds = [(-1, 1), (-1, 1), (-2, 2), (0, 2 * 24 * 60 ** 2)]
    
    gen_algo = GA(get_fitness, bounds, 42)
        
    gen_algo.evolve_population(20, 80, 6, 0.6, 4, 6, path_to_save_data=hf.sim_data_dir + "/custom_genetic_algo/test_run2")
    
    
    
    pass
    
        
    
        
    
    
    
    
    
    