import numpy as np
import itertools
import math


class GA:
    def __init__(
        self,
        fitness_function,
        initial_genes_array,
        genes_array_boundaries,
        seed="random",
        ) -> None:
                
        self.fitness_function = fitness_function
        self.initial_genes_array = initial_genes_array
        self.genes_array_boundaries = genes_array_boundaries
        if seed != "random":
            self.np_rand = np.random.RandomState(seed)
        else:
            self.np_rand = np.random.RandomState()
            
    
    def select_best_mates(self, no_mates):    
            
        sorted_ids = np.argsort(self.population_fitness)
                
        self.mating_genes = []
        self.mating_fitnie = []
        
        for i in range(no_mates):
            self.mating_genes.append(self.gene_pool[sorted_ids[i]])
            self.mating_fitnie.append(self.fitness_pool[sorted_ids[i]])
        
    def evaluate_new_genes(self, children_genes):
        
        children_fitness = []
        
        for child in children_genes:
            
            children_fitness.append(self.fitness_function(child))
            
        return children_fitness
    
    def generate_children(self, no_mates, no_children, mutation_probability, no_best_parents_to_keep):
        
        self.select_best_mates(self, no_mates)
        
        # breed        
        fuck_list = list(itertools.combinations_with_replacement(np.arange(0, 5), 2))
        fuck_list = [c for c in fuck_list if c[0] != c[1]]
        frac = no_children / len(fuck_list)
        fuck_list = fuck_list * math.ceil(frac)
        
        children_genes = []
        
        for i in range(no_children):
        
            couple = fuck_list[i]
            
            gene_selection = np.random.randint(2, size=len(self.initial_genes_array))
            new_child = []
            
            for j, parent_id in enumerate(gene_selection):
                if np.random.random() < mutation_probability: 
                    new_gene = np.random.uniform(self.genes_array_boundaries[j][0], self.genes_array_boundaries[j][1])
                else:
                    new_gene = self.mating_genes[couple[parent_id]][j]
                    
                new_child.append(new_gene)
                
                pass
                
            children_genes.append(new_child)
        
        children_fitness = self.evaluate_new_genes(self, children_genes)
        
        children_sorted_ids = np.argsort(children_fitness)
        parents_sorted_ids = np.argsort(self.population_fitness)
        
        if no_best_parents_to_keep > 0:
            for i in range(no_best_parents_to_keep):
                if children_fitness[children_sorted_ids[no_children - i]]
        
        
    
    def evolve_population(
        self, 
        individuals, 
        generations, 
        no_mates, 
        mutation_probability,
        max_genes_to_mutate,
        no_best_parents_to_keep,
        mp_nodes=None):
        
        
        
        
        
        
        
        
        pass
        
        
if __name__ == "__main__":
    
    
    print(math.ceil(0.001))
    print(math.ceil(1.001))
    
    
    
    pass
    
        
    
        
    
    
    
    
    
    