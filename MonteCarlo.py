import pdb
import sys
from pprint import pprint
import numpy as np
import os
import multiprocessing as mp
from time import time
from scipy.stats.qmc import Sobol       # for Sobol's sampling method
from scipy.stats import qmc
import helper_functions as hf

from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import environment
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.math import interpolators
import tudatpy.util as util
import simulation as sim

def make_log(log_list, no_runs):
    def logger(evaluation):
        log_list.append(0)
        print(f"\t {len(log_list)} / {no_runs} completed, {no_runs - len(log_list)} remaining \t\t", end="\r")
    return logger


def convert_mp_output_to_dict(tup, di):
    di = dict(tup)
    return di


def monte_carlo_entry(current_param_dict, sim_idx, savepath):
    return sim.run_simulation(savepath,
                              6 * 31 * 24 * 60**2,
                              decision_variable_dic=current_param_dict,
                              sim_idx=sim_idx)


""" WARNING, RUNNING THIS FILE WILL GENERATE ~10GB OF DATA"""
if __name__ == '__main__':
    design_space_method = 'monte_carlo'
    # design_space_method = 'monte_carlo_one_at_a_time'
    write_results_to_file = True

    augustas = False
    multiprocessing = True

    random_seed = 42
    cores_to_use = mp.cpu_count() - 2
    output_path = hf.external_sim_data_dir + f'/DesignSpace/{design_space_method}/'

    decision_parameter_range = [[-0.5, -0.5, -0.5, 0], [0.5, 0.5, 0.5, 15*24*60*60]]

    if design_space_method == 'monte_carlo_one_at_a_time':
        number_of_simulations_per_parameter = 2**7
        number_of_simulations = 4 * number_of_simulations_per_parameter
        nominal_parameters = list(sim.default_decision_variable_dic.values())
        # Don't question this, it works
        nominal_parameters = [nominal_parameters[0], nominal_parameters[1][0], nominal_parameters[1][1],
                              nominal_parameters[1][2], nominal_parameters[2]]
        np.random.seed(random_seed)  # Slightly outdated way of doing this, but works
        print('\n Random Seed :', random_seed, '\n')

        sobol_sequence = Sobol(d=1, seed=42)
        m = int(np.log2(number_of_simulations_per_parameter))
        # THE SAME sequence of numbers between 0-1 used to scale each parameter separately
        fractional_variation_sample = sobol_sequence.random_base2(m)
    else:
        number_of_simulations = 2**14
        np.random.seed(random_seed)  # Slightly outdated way of doing this, but works
        print('\n Random Seed :', random_seed, '\n')

        sobol_sequence = Sobol(d=4, seed=42)
        m = int(np.log2(number_of_simulations))
        sequences = sobol_sequence.random_base2(m=m)

    parameters = dict()
    objectives_and_constraints = dict()

    for sim_index in range(number_of_simulations):
        if design_space_method == 'monte_carlo_one_at_a_time':
            decision_parameters = nominal_parameters.copy()

            current_parameter = int(sim_index / number_of_simulations_per_parameter)


            # Find range over which to scale parameter using Sobol array defined previously
            current_param_range = decision_parameter_range[1][current_parameter] - \
                                  decision_parameter_range[0][current_parameter]

            # Edit current parameter
            current_variation_index = sim_index - (current_parameter * number_of_simulations_per_parameter)

            # Transform variation from rsw
            variation = (fractional_variation_sample[current_variation_index] * current_param_range)[0]

            decision_parameters[current_parameter] = decision_parameter_range[0][current_parameter] + variation

            dv = [decision_parameters[0], decision_parameters[1], decision_parameters[2]]
            # Transform to input dictionary
            decision_parameters_dict = {'dv': dv,
                                        't_impulse': decision_parameters[3]}

        else:
            decision_parameters = np.array(decision_parameter_range[0]) + sequences[sim_index, :] * np.array(np.subtract(
                decision_parameter_range[1], decision_parameter_range[0]))

            # Transform to input dictionary
            dv = [decision_parameters[0], decision_parameters[1], decision_parameters[2]]

            decision_parameters_dict = {'dv': dv,
                                        't_impulse': decision_parameters[3]}

        parameters[sim_index] = decision_parameters_dict
        
    if augustas:

        with mp.get_context("spawn").Pool(cores_to_use, maxtasksperchild=8) as pool:
            inputs = []
            for run_idx, parameter_set in enumerate(list(parameters.values())):
                inputs.append((parameter_set, run_idx, output_path + f"/iter_{run_idx}"))

            sim_indices = []
            outputs = pool.starmap(monte_carlo_entry, inputs)
            
            # Process outputs to dict form
            objectives_and_constraints = {}
            objectives_and_constraints = convert_mp_output_to_dict(outputs, objectives_and_constraints)    
    
    else:
        input_lst = []
        for run_idx, parameter_set in enumerate(list(parameters.values())):
            input_lst.append([parameter_set, run_idx, output_path + f"/iter_{run_idx}"])
        
        objectives_and_constraints = {}
        async_results = []
        log_list = []
        if multiprocessing:    
            pool = mp.Pool(cores_to_use, maxtasksperchild=40)
            for inp in input_lst:
                async_results.append(pool.apply_async(monte_carlo_entry, args=inp, callback=make_log(log_list, len(input_lst))))
            pool.close()
            pool.join() 
            
            results = []
            for async_res in async_results:
                results.append(async_res.get())
            
            objectives_and_constraints = convert_mp_output_to_dict(results, objectives_and_constraints)    
                
        else:
            results = []
            for i, inp in enumerate(input_lst):
                results.append(monte_carlo_entry(*inp))
                objectives_and_constraints = convert_mp_output_to_dict(results, objectives_and_constraints)    
                print(f"{i} / {len(input_lst)} completed \t\t", end="\r")

    if write_results_to_file:
        parameters_for_json = {}
        for key, value in parameters.items():
            parameters_for_json[key] = hf.make_dic_safe_for_json(value)
        hf.save_dict_to_json(parameters_for_json, output_path + "/parameter_values.dat")
        save2txt(objectives_and_constraints, 'objectives_constraints.dat', output_path)