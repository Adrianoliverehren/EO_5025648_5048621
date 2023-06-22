import numpy as np




class GA:
    def __init__(
        self,
        fitness_function,
        initial_genes_array,
        genes_array_boundaries,
        mutation_probability,
        max_genes_to_mutate,
        seed="random",
        ) -> None:
                
        self.fitness_function = fitness_function
        self.initial_genes_array = initial_genes_array
        self.genes_array_boundaries = genes_array_boundaries
        self.mutation_probability = mutation_probability
        self.max_genes_to_mutate = max_genes_to_mutate
        if seed != "random":
            self.np_rand = np.random.RandomState(seed)
        else:
            self.np_rand = np.random.RandomState()
        
    def evolve_population(self, individuals, generations, mp_nodes=None):
        
        
        
        
        pass
        
        
    
        
    
    
    
    
    
    