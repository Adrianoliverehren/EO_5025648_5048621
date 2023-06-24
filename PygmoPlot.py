import os
import pickle

import matplotlib.pyplot as plt
import pygmo as pg


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

    # Plot fitness over time
    fig = plt.figure()
