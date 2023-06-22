import pdb

from scipy.optimize import minimize
from simulation import run_simulation
import numpy as np
import multiprocessing as mp
from scipy.stats.qmc import Sobol
import helper_functions as hf


def sim_wrapper(decision_var_arr):
    """Wraps simulation function to allow for input and output format necessary for scipy optimization algorithms"""

    decision_var_dict = {'dv': np.array([decision_var_arr[0], decision_var_arr[1], decision_var_arr[2]]),
                         't_impulse': decision_var_arr[3]}

    _, [unpenalized_objective, constraint] = run_simulation(False, 6 * 31 * 24 * 60**2,
                                                            decision_variable_dic=decision_var_dict)

    if constraint > 0:
        penalty = constraint**2 * 10**5
    else:
        penalty = 0
    objective = -unpenalized_objective + penalty
    return objective


def optimize_w_scipy(alg_name):
    """ Function defining full optimization problem with given algorithm"""
    # Settings
    popsize = 2
    max_generations = 25
    bounds = [(-1, 1), (-1, 1), (-2, 2), (0, 2 * 24 * 60 ** 2)]

    # Initialize population
    sobol_sequence = Sobol(d=4, seed=42)
    sequences = sobol_sequence.random_base2(m=int(np.log2(popsize)))
    result_obj_lst = []
    for pop_idx in range(popsize):
        dvr = bounds[0][0] + sequences[pop_idx][0] * (bounds[0][1] - bounds[0][0])
        dvs = bounds[1][0] + sequences[pop_idx][1] * (bounds[1][1] - bounds[1][0])
        dvw = bounds[2][0] + sequences[pop_idx][2] * (bounds[2][1] - bounds[2][0])
        t_impulse = bounds[3][0] + sequences[pop_idx][3] * (bounds[3][1] - bounds[3][0])
        initial_pop = np.array([dvr, dvs, dvw, t_impulse])

        # Select alg and optimize
        if alg_name == 'NMS':
            result_obj_lst.append(minimize(fun=sim_wrapper, x0=initial_pop, method='Nelder-Mead', bounds=bounds,
                                  options={'maxiter': max_generations}))
        elif alg_name == 'BFGS':
            result_obj_lst.append(minimize(fun=sim_wrapper, x0=initial_pop, method='BFGS', bounds=bounds,
                                  options={'maxiter': max_generations}))

    return result_obj_lst


if __name__ == '__main__':
    cores_to_use = mp.cpu_count() - 2
    alg_lst = [['NMS']
               # , ['BFGS']
               ]

    savedirs = [f'./opt_output_{alg_name[0]}/hi' for alg_name in alg_lst]
    for savedir in savedirs:
        hf.make_ALL_folders_for_path(savedir)

    plot = True
    with mp.get_context("spawn").Pool(cores_to_use, maxtasksperchild=1) as pool:
        outputs = pool.starmap(optimize_w_scipy, alg_lst)
        print(outputs)

    for alg_name, output in zip(alg_lst, outputs):
        print(output)
        filepath = f'./opt_output_{alg_name[0]}/'
        for gen_idx, save_obj in enumerate(output):
            filename = f'gen_{gen_idx}'
            np.save(filepath + filename, output)

    if plot:
        for alg_idx, alg in enumerate(alg_lst):
            pass


