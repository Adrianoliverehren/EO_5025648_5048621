import os
import pickle
import sys

import matplotlib.pyplot as plt
import pygmo as pg
import numpy as np
from simulation import run_simulation
import helper_functions as hf


def save_stuff(data_dir, save_dir, filename):
    """
    Takes data of best entries for each generation in a given directory and saves it compactly
    Args:
        data_dir: directory with all generation files
        save_dir: directory to save to
        filename: name of saved file
    """
    print('Processing', data_dir)
    fitness_lst = []
    t_survive_lst = []
    constraint_lst = []
    x_lst = []
    generation_lst = []
    # For every generation read the population object
    for gen_idx in range(len(os.listdir(data_dir))):
        current_subdir = data_dir + f'gen_{gen_idx}/'
        with open(current_subdir + 'population.pkl', 'rb') as f:
            current_population = pickle.load(f)

            # Get fitness
            fitness_lst.append(current_population.get_f()[current_population.best_idx()])

            # get x and repropagate to get t survive and constraint value
            best_x_of_gen = current_population.get_x()[current_population.best_idx()]
            decision_var_dict = {'dv': np.array([best_x_of_gen[0], best_x_of_gen[1], best_x_of_gen[2]]),
                                 't_impulse': best_x_of_gen[3]}

            _, [unpenalized_objective, constraint] = run_simulation(False, 6 * 31 * 24 * 60 ** 2,
                                                           decision_variable_dic=decision_var_dict)

            t_survive_lst.append(unpenalized_objective)

            constraint_lst.append(constraint)

            x_lst.append(best_x_of_gen)
            generation_lst.append(gen_idx)

    save_matrix = [fitness_lst, t_survive_lst, constraint_lst, x_lst, generation_lst]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir + filename, 'wb') as outp:
        pickle.dump(save_matrix, outp, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    read_dir = './NMS/'
    save_dir = './Figures/Pygmo/NMS/'
    hf.make_ALL_folders_for_path(save_dir + 'hello')

    hf.make_ALL_folders_for_path('./OutputStuff/' + 'hello')
    save_stuff('./GACO/', './OutputStuff/', 'best_of_GACO.pkl')
    save_stuff('./PSO/', './OutputStuff/', 'best_of_PSO.pkl')
    save_stuff('./DE/', './OutputStuff/', 'best_of_DE.pkl')

    plot = False
    if plot:
        print('plotting')
        all_generations = []
        # For every generation read the population object
        for gen_idx in range(len(os.listdir(read_dir))):
            current_subdir = read_dir + f'gen_{gen_idx}/'
            with open(current_subdir + 'population.pkl', 'rb') as f:
                current_population = pickle.load(f)
            all_generations.append(current_population)

        # Plot fitness over generations
        if True:
            fig = plt.figure(11, figsize=(4, 4))
            ax1 = fig.add_subplot(111)

            gen_lst = []
            best_fit_lst = []
            for gen_idx, generation in enumerate(all_generations):
                # Plot fitness vs generation
                gen_lst.append(gen_idx)
                best_fit_lst.append(generation.get_f()[generation.best_idx()])
            ax1.plot(gen_lst, np.array(best_fit_lst)/(24 * 60**2))

            ax1.set_xlabel('Generation [-]')
            ax1.set_ylabel('Best Fitness [days]')
            ax1.grid()
            plt.tight_layout()
            plt.savefig(save_dir + 'fitness_v_gen.pdf', bbox_inches='tight')

        # Plot survival time over generations
        if True:
            fig = plt.figure(12, figsize=(4, 4))
            ax1 = fig.add_subplot(111)

            gen_lst = []
            t_survive_lst = []
            for gen_idx, generation in enumerate(all_generations):
                # Plot fitness vs generation
                gen_lst.append(gen_idx)
                best_x_of_gen = generation.get_x()[generation.best_idx()]

                decision_var_dict = {'dv': np.array([best_x_of_gen[0], best_x_of_gen[1], best_x_of_gen[2]]),
                                     't_impulse': best_x_of_gen[3]}

                _, [unpenalized_objective, _] = run_simulation(False, 6 * 31 * 24 * 60 ** 2,
                                                               decision_variable_dic=decision_var_dict)

                t_survive_lst.append(unpenalized_objective)

            ax1.plot(gen_lst, np.array(t_survive_lst)/(24 * 60**2))

            ax1.set_xlabel('Generation [-]')
            ax1.set_ylabel('Best Surival time [days]')
            ax1.grid()
            plt.tight_layout()
            plt.savefig(save_dir + 't_survive_v_gen.pdf', bbox_inches='tight')

