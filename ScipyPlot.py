import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import helper_functions as hf


def fitness_to_time(fitness, constraint):
    time = []
    for index in range(len(fitness)):
        current_constraint = constraint[index]
        current_fitness = fitness[index]
        if current_constraint > 0:
            time.append(current_constraint**2 * 10**6 - current_fitness)

        else:
            time.append(-current_fitness)

    return time


def best_pop_idx(result_obj_lst):
    best_idx = -1
    best_fit = np.inf
    for idx, result_obj in enumerate(result_obj_lst):
        if result_obj.fun < best_fit:
            best_fit = result_obj.fun
            best_idx = idx

    return best_idx


if __name__ == '__main__':
    # Directory to read npy files from for plotting
    read_dir = './opt_output_Nelder-Mead/'
    plot_dir = './Figures/Scipy/NMS/'

    hf.make_ALL_folders_for_path(plot_dir + 'hello')

    result_obj_lst = []
    fitness_history_lst = []
    constraint_history_lst = []
    n_pops = int(len(os.listdir(read_dir))/3)

    for pop_idx in range(n_pops):
        scipy_file = f'pop_{pop_idx}_out'
        fitness_file = f'pop_{pop_idx}_fit'
        constraint_file = f'pop_{pop_idx}_con'

        with open(read_dir + scipy_file, 'rb') as f:
            result_obj = pickle.load(f)
        result_obj_lst.append(result_obj)

        with open(read_dir + fitness_file, 'rb') as f:
            fitness = pickle.load(f)
        fitness_history_lst.append(fitness)

        with open(read_dir + constraint_file, 'rb') as f:
            constraint = pickle.load(f)
        constraint_history_lst.append(constraint)



    pops_to_plot = [best_pop_idx(result_obj_lst)]
    fig = plt.figure(11, figsize=(4, 4))
    ax1 = fig.add_subplot(111)
    for pop_idx in pops_to_plot:
        # Plot fitness vs generation
        gen_lst = np.arange(0, len(fitness_history_lst[pop_idx]), 1)
        ax1.plot(gen_lst, np.array(fitness_history_lst[pop_idx])/(24 * 60**2), label=f'Pop #{pop_idx}')

    ax1.set_xlabel('Generation [-]')
    ax1.set_ylabel('Fitness [days]')
    ax1.legend()
    ax1.grid()
    plt.tight_layout()
    plt.savefig(plot_dir + 'fitness_v_gen.pdf', bbox_inches='tight')

    fig = plt.figure(22, figsize=(4, 4))
    ax1 = fig.add_subplot(111)
    for pop_idx in pops_to_plot:
        # Plot constraint vs generation
        gen_lst = np.arange(0, len(constraint_history_lst[pop_idx]), 1)
        ax1.plot(gen_lst, np.array(constraint_history_lst[pop_idx]), label=f'Pop #{pop_idx}')

    ax1.set_xlabel('Generation [-]')
    ax1.set_ylabel('Constraint value [s]')
    ax1.legend()
    ax1.grid()
    plt.tight_layout()
    plt.savefig(plot_dir + 'con_v_gen.pdf', bbox_inches='tight')

    fig = plt.figure(33, figsize=(4, 4))
    ax1 = fig.add_subplot(111)
    for pop_idx in pops_to_plot:
        # Plot constraint vs generation
        gen_lst = np.arange(0, len(constraint_history_lst[pop_idx]), 1)
        ax1.plot(gen_lst,
                 np.array(fitness_to_time(fitness_history_lst[pop_idx], constraint_history_lst[pop_idx])) / (24*60*60),
                 label=f'Pop #{pop_idx}')

    ax1.set_xlabel('Generation [-]')
    ax1.set_ylabel('Survival time [s]')
    ax1.legend()
    ax1.grid()
    plt.tight_layout()
    plt.savefig(plot_dir + 't_survive_v_gen.pdf', bbox_inches='tight')

    print('Best decision values = ', result_obj_lst[best_pop_idx(result_obj_lst)].x)
    plt.show()