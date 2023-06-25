from PygmoOpt import optimize
import pygmo as pg

if __name__ == '__main__':

    harmonics_to_investigate = [(2, 2),
                                (4, 4),
                                (8, 8),
                                (10, 10),
                                (20, 20),
                                (40, 40)]

    save_dirs = [f'./EnvironmentInv/{harmonic}/' for harmonic in harmonics_to_investigate]
    n_generations = 64
    pop_size = 32
    algo = pg.de()

    x_min_lst = [-1, -1, -2, 120]
    x_max_lst = [1, 1, 2, 2 * 24 * 60 ** 2]

    for harmonics, save_dir in zip(harmonics_to_investigate, save_dirs):
        print(harmonics, ' Running...')
        optimize(save_dir, algo, x_min_lst, x_max_lst, n_generations, pop_size, seed=42,
                 BFE=False, spherical_harmonics=harmonics)