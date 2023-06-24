import pygmo as pg
import numpy as np
import multiprocessing as mp
from simulation import run_simulation


class GEOProblem:
    def __init__(self, xmin_arr, xmax_arr):
        self.xmin_arr = xmin_arr
        self.xmax_arr = xmax_arr
        self.n_cores = mp.cpu_count() - 2

    def get_bounds(self):
        bounds = (self.xmin_arr, self.xmax_arr)
        return bounds

    def get_nobj(self):
        return 1

    @staticmethod
    def fitness(design_parameters):
        decision_var_dict = {'dv': np.array([design_parameters[0], design_parameters[1], design_parameters[2]]),
                             't_impulse': design_parameters[3]}

        _, [unpenalized_objective, constraint] = run_simulation(False, 6 * 31 * 24 * 60 ** 2,
                                                                decision_variable_dic=decision_var_dict)

        if constraint > 0:
            penalty = constraint ** 2 * 10 ** 6
        else:
            penalty = 0
        objective = -unpenalized_objective + penalty

        return [objective]

    def batch_fitness(self, design_parameter_vectors):
        len_single_vector = 4
        # Reshape from 1xn*m to mxn
        dpv_lst = design_parameter_vectors.reshape(len(design_parameter_vectors) // len_single_vector, len_single_vector)

        inputs, fitnesses = [], []
        with mp.get_context("spawn").Pool(self.n_cores) as pool:
            outputs = pool.map(self.fitness, inputs)

        for output in outputs:
            fitnesses.append(output)

        return fitnesses



