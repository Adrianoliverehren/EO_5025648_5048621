import multiprocessing as mp

import pygmo as pg
import numpy as np

import platform
import pickle
import os

from PygmoProblem import GEOProblem
from helper_functions import make_ALL_folders_for_path

if __name__ == '__main__':
    # Inputs
    x_min_lst = [-1, -1, -2, 120]
    x_max_arr = [1, 1, 2, 2 * 24 * 60 ** 2]
    n_generations = 45
    pop_size = 64
    seed = 42
    BFE = False  # set to True for parallel processing, does not work with some algorithms
    save_dir = './NMS/'
    # save_dir = './GACO/'
    # save_dir =  './PSO/'

    # Pick algorithm
    # Nelder-Mead Simplex
    algo = pg.nlopt(solver="neldermead")
    # Extended ant colony
    # algo = pg.gaco()
    # Particle Swarm
    # algo = pg.pso()

    # Initialize problem
    problem = pg.problem(GEOProblem(x_min_lst, x_max_arr))

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

