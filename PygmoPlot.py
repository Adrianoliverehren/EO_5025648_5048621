import os
import pickle

import matplotlib.pyplot as plt
import pygmo as pg
import numpy as np
from simulation import run_simulation


if __name__ == '__main__':
    read_dir = './PSO/'
    save_dir = './Figures/Pygmo/PSO/'

    all_generations = []
    # For every generation read the population object
    for gen_idx in range(len(os.listdir(read_dir))):
        current_subdir = read_dir + f'gen_{gen_idx}/'
        with open(read_dir + 'population.pkl', 'rb') as f:
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
        ax1.legend()
        ax1.grid()
        plt.tight_layout()
        plt.savefig(save_dir + 'fitness_v_gen.pdf', bbox_inches='tight')


    if True:
        # Plot survival time over generations
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
        ax1.legend()
        ax1.grid()
        plt.tight_layout()
        plt.savefig(save_dir + 't_survive_v_gen.pdf', bbox_inches='tight')