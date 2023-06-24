import multiprocessing as mp

import pygmo as pg
import numpy as np

import platform
import pickle
import os

from PygmoProblem import GEOProblem
from helper_functions import make_ALL_folders_for_path


def optimize(save_dir, algo, xmin, xmax, n_generations, pop_size, seed=42, BFE=False):
    # Initialize problem
    problem = pg.problem(GEOProblem(xmin, xmax))

    # Turn on parallel processing and create algorithm obj
    if BFE:
        algo.set_bfe(pg.bfe())
    algo = pg.algorithm(algo)

    # Initialize population
    bfe_pop = pg.default_bfe() if BFE else None
    if BFE:
        pop = pg.population(prob=problem, size=pop_size, seed=seed, b=bfe_pop)
    else:
        pop = pg.population(prob=problem, size=pop_size, seed=seed)

    # Save objectives and parameters of initial generation
    current_dir = save_dir + f'gen_0/'
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)

    np.savetxt(current_dir + 'fitness.dat', pop.get_f())
    np.savetxt(current_dir + 'decisions.dat', pop.get_x())
    # Save pop object
    with open(current_dir + 'population.pkl', 'wb') as outp:
        pickle.dump(pop, outp, pickle.HIGHEST_PROTOCOL)

    # If run on Windows, hold
    if platform.system()[0] == 'W':
        mp.freeze_support()

    # Run algorithm over generations
    for gen_i in range(n_generations):
        print(f'Evolution: {gen_i + 1}/{n_generations}')
        pop = algo.evolve(pop)

        # save objectives and decision params of current generation
        current_dir = save_dir + f'gen_{gen_i + 1}/'

        if not os.path.exists(current_dir):
            os.makedirs(current_dir)
        np.savetxt(current_dir + 'fitness.dat', pop.get_f())
        np.savetxt(current_dir + 'decisions.dat', pop.get_x())

        # Save pop object
        with open(current_dir + 'population.pkl', 'wb') as outp:
            pickle.dump(pop, outp, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # Inputs
    x_min_lst = [-1, -1, -2, 120]
    x_max_lst = [1, 1, 2, 2 * 24 * 60 ** 2]
    n_generations = 65
    pop_size = 128
    seed = 42
    BFE = False  # set to True for parallel processing
    # save_dir = './NMS/'
    # save_dir = './GACO/'
    # save_dir = './PSO/'
    save_dir = './DE_tuned/'
    # save_dir = './BFGS/'

    # Pick algorithm
    # Nelder-Mead Simplex
    # algo = pg.nlopt(solver="neldermead")

    # Extended ant colony   -> needs 40? evols
    # algo = pg.gaco()

    # Particle Swarm    -> needs 65 evols
    # algo = pg.pso()

    # Differential evolution    -> needs 65 evols
    algo = pg.de(F=0.4, CR=1.0)  # 3 settings, F=float (weight coeff), CR=float (crossover prob), variant=int (mutation variant)

    # BFGS
    # algo = pg.nlopt(solver='lbfgs')

    investigate_settings = False

    if investigate_settings:
        from itertools import product as combine

        n_generations = 65
        pop_size = 32   # Nominal was 128

        F_range = np.arange(0.8, 1.01, 0.2)
        CR_range = np.arange(0.4, 1.01, 0.2)
        algo_setting_lst = [(round(F, 2), round(CR, 2)) for (F, CR) in list(combine(F_range, CR_range))]

        save_dirs = [f'./DE_settings/F_{F}_CR_{CR}/' for (F, CR) in algo_setting_lst]

        for save_dir, algo_setting in zip(save_dirs, algo_setting_lst):
            print(save_dir, ' Running.....')
            algo = pg.de(F=algo_setting[0], CR=algo_setting[1])
            optimize(save_dir, algo, x_min_lst, x_max_lst, n_generations, pop_size, seed=42, BFE=False)

    else:
        optimize(save_dir, algo, x_min_lst, x_max_lst, n_generations, pop_size, seed=42, BFE=BFE)
