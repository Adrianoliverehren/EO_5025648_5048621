import gc
import pdb
import sys

from scipy.optimize import minimize
from simulation import run_simulation
import numpy as np
import multiprocessing as mp
import pickle
from scipy.stats.qmc import Sobol
import helper_functions as hf


class ObjHistory():
    def __init__(self):
        self.objective_history = []
        self.decision_history = []
        self.constraint_history = []
        self.true_fitness_history = []
        self.true_constraint_history = []


def sim_wrapper(decision_var_arr, ObjectiveHistory_obj):
    """Wraps simulation function to allow for input and output format necessary for scipy optimization algorithms"""

    decision_var_dict = {'dv': np.array([decision_var_arr[0], decision_var_arr[1], decision_var_arr[2]]),
                         't_impulse': decision_var_arr[3]}

    _, [unpenalized_objective, constraint] = run_simulation(False, 6 * 31 * 24 * 60**2,
                                                            decision_variable_dic=decision_var_dict)

    if constraint > 0:
        penalty = constraint**2 * 10**6
    else:
        penalty = 0
    objective = -unpenalized_objective + penalty

    ObjectiveHistory_obj.objective_history.append(objective)
    ObjectiveHistory_obj.constraint_history.append(constraint)
    ObjectiveHistory_obj.decision_history.append(decision_var_arr)
    return objective


def optimize_w_scipy(alg_name):
    """ Function defining full optimization problem with given algorithm"""
    # Settings
    popsize = 64
    max_generations = 50
    bounds = [(-1, 1), (-1, 1), (-2, 2), (0, 2 * 24 * 60 ** 2)]

    # Initialize population
    sobol_sequence = Sobol(d=4, seed=42)
    sequences = sobol_sequence.random_base2(m=int(np.log2(popsize)))
    result_obj_lst = []
    all_pop_fitness_history = []
    all_pop_constraint_history = []

    for pop_idx in range(popsize):
        dvr = bounds[0][0] + sequences[pop_idx][0] * (bounds[0][1] - bounds[0][0])
        dvs = bounds[1][0] + sequences[pop_idx][1] * (bounds[1][1] - bounds[1][0])
        dvw = bounds[2][0] + sequences[pop_idx][2] * (bounds[2][1] - bounds[2][0])
        t_impulse = bounds[3][0] + sequences[pop_idx][3] * (bounds[3][1] - bounds[3][0])
        initial_pop = np.array([dvr, dvs, dvw, t_impulse])

        # Select alg and optimize
        if True:
            print('Evaluating pop ', pop_idx + 1, '/', popsize)

            current_objective_history = ObjHistory()

            # Callback function to save fitness history
            def save_fitness(input):
                # Find which function evaluation is actually necessary
                for index, decision_vec in enumerate(current_objective_history.decision_history):
                    list_entries_match = [True if stored_x == input_x else False for stored_x, input_x in zip(decision_vec, input)]
                    if all(list_entries_match):
                        true_index = index

                # Append it to list of values to save
                current_objective_history.true_fitness_history.append(current_objective_history.objective_history[true_index])
                current_objective_history.true_constraint_history.append(current_objective_history.constraint_history[true_index])

            current_objective_history.constraint_history = []
            current_objective_history.objective_history = []
            current_objective_history.decision_history = []
            min_func = lambda func, args, x0, method, bounds, options, callback: minimize(fun=func, args=args,
                                                                                          x0=x0, method=method,
                                                                                          bounds=bounds,
                                                                                          options=options,
                                                                                          callback=callback)

            result_obj_lst.append(min_func(sim_wrapper, (current_objective_history, ), initial_pop, alg_name,
                                           bounds, {'maxiter': max_generations, 'return_all': True},
                                           save_fitness))

            del min_func
            gc.collect()
            """ 
            result_obj_lst.append(minimize(fun=sim_wrapper, args=(current_objective_history,),
                                           x0=initial_pop, method='Nelder-Mead', bounds=bounds,
                                  options={'maxiter': max_generations, 'return_all': True}, callback=save_fitness))
            """
            # Append fitness and constraint values to save arrays

            all_pop_fitness_history.append(current_objective_history.true_fitness_history)
            all_pop_constraint_history.append(current_objective_history.true_constraint_history)

    return result_obj_lst, all_pop_fitness_history, all_pop_constraint_history


if __name__ == '__main__':
    cores_to_use = mp.cpu_count() - 2
    alg_lst = [
        'Nelder-Mead'
        # 'L-BFGS-B'
               ]

    savedirs = [f'./opt_output_{alg_name}/hi' for alg_name in alg_lst]
    for savedir in savedirs:
        hf.make_ALL_folders_for_path(savedir)

    opt_outputs = []
    fitness_outputs = []
    constraint_outputs = []
    for alg in alg_lst:
        opt_output, fitness_output, constraint_output = optimize_w_scipy(alg)
        opt_outputs.append(opt_output)
        fitness_outputs.append(fitness_output)
        constraint_outputs.append(constraint_output)

    # Save
    # output = list of scipy output objects for each pop
    # fitness_histories = list of custom objects with fitness history for each pop
    for alg_name, output, fitness_histories, constraint_histories in zip(alg_lst, opt_outputs,
                                                                         fitness_outputs, constraint_outputs):
        filepath = f'./opt_output_{alg_name}/'
        for pop_idx, (save_obj, fitness, constraint) in enumerate(zip(output, fitness_histories, constraint_histories)):
            filename_output = f'pop_{pop_idx}_out'
            filename_fitness = f'pop_{pop_idx}_fit'
            filename_constraint = f'pop_{pop_idx}_con'

            hf.make_ALL_folders_for_path(filepath + filename_output)
            with open(filepath + filename_output, 'wb') as outp:
                pickle.dump(save_obj, outp, pickle.HIGHEST_PROTOCOL)

            hf.make_ALL_folders_for_path(filepath + filename_fitness)
            with open(filepath + filename_fitness, 'wb') as outp:
                pickle.dump(fitness, outp, pickle.HIGHEST_PROTOCOL)

            hf.make_ALL_folders_for_path(filepath + filename_constraint)
            with open(filepath + filename_constraint, 'wb') as outp:
                pickle.dump(constraint, outp, pickle.HIGHEST_PROTOCOL)
