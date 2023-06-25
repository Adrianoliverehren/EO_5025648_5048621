from PygmoOpt import optimize
from simulation import run_simulation
import helper_functions as hf
import matplotlib.pyplot as plt
import pygmo as pg
import numpy as np
import pickle
import os


def env_design_domain(champion_x_dict, save_dir):
    # init plot
    figure = plt.figure(123, figsize=(4, 4))
    orig_map = plt.cm.get_cmap('jet')
    # reversing the original colormap using reversed() function
    reversed_map = orig_map.reversed()

    # Plot each dV component
    marker_lst = ['.', '+', 'x']
    label_lst = [r'$\Delta \bar{V}_R$', r'$\Delta \bar{V}_S$', r'$\Delta \bar{V}_W$']
    for dv_component_idx in range(3):
        # Construct arrays to input for plotting
        y_array = []
        env_arr = []
        t_impulse_arr = []
        for env in champion_x_dict.keys():
            y_array.append(champion_x_dict[env][dv_component_idx])
            env_arr.append(str(env))
            t_impulse_days = champion_x_dict[env][-1] / (24 * 60 * 60)
            t_impulse_arr.append(t_impulse_days)

        plt.scatter(env_arr, y_array, c=t_impulse_arr, cmap=reversed_map, marker=marker_lst[dv_component_idx],
                    s=10, label=label_lst[dv_component_idx])


    colorbar = plt.colorbar()
    colorbar.set_label('Time of Impulse [Days]')
    plt.grid()
    plt.tight_layout()
    hf.make_ALL_folders_for_path(save_dir + 'hello')
    plt.savefig('./save_dir/x_opt_env.pdf', bbox_inches='tight')


if __name__ == '__main__':
    run = False
    plot = True
    read_dir = './EnvironmentInv/'
    plot_dir = './Figures/Environment/'
    harmonics_to_investigate = [(2, 2),
                                (4, 4),
                                (8, 8),
                                (10, 10),
                                (20, 20),
                                (40, 40)]
    if run:
        save_dirs = [f'./EnvironmentInv/{harmonics_to_investigate[0]}/']
        n_generations = 64
        pop_size = 32
        algo = pg.de()

        x_min_lst = [-1, -1, -2, 120]
        x_max_lst = [1, 1, 2, 2 * 24 * 60 ** 2]

        for harmonics, save_dir in zip(harmonics_to_investigate, save_dirs):
            optimize(save_dir, algo, x_min_lst, x_max_lst, n_generations, pop_size, seed=42,
                     BFE=False, spherical_harmonics=harmonics)

    if plot:
        # Get best design from each environment setting
        champ_x_dict = dict()
        champ_f_dict = dict()
        for env_dir in os.listdir(read_dir):
            final_gen_idx = len(os.listdir(read_dir + env_dir + '/')) - 1
            current_subdir = read_dir + env_dir + '/' + f'gen_{final_gen_idx}/'
            with open(current_subdir + 'population.pkl', 'rb') as f:
                current_population = pickle.load(f)
                champ_x_dict[env_dir] = current_population.champion_x
                champ_f_dict[env_dir] = current_population.chamption_f

        # Plot survival time
        fig = plt.figure(11, figsize=(4, 4))
        ax1 = fig.add_subplot(111)

        for env in champ_f_dict.keys():
            best_x_of_env = champ_x_dict[env]
            decision_var_dict = {'dv': np.array([best_x_of_env[0], best_x_of_env[1], best_x_of_env[2]]),
                                 't_impulse': best_x_of_env[3]}

            _, [unpenalized_objective, _] = run_simulation(False, 6 * 31 * 24 * 60 ** 2,
                                                           decision_variable_dic=decision_var_dict,
                                                           spherical_harmonics=env)

            ax1.bar(str(env), unpenalized_objective / (24 * 60 ** 2))

        ax1.set_ylabel('Best Survival Time [days]')
        ax1.set_ylim(ax1.get_ylim()[::-1])
        ax1.grid(axis='y')
        ax1.set_xticks(ax1.get_xticks(), ax1.get_xticklabels(), rotation=90)
        plt.tight_layout()
        plt.savefig(plot_dir + 't_survive_v_env.pdf', bbox_inches='tight')

        # Plot optimal designs
        env_design_domain(champ_x_dict, plot_dir)

