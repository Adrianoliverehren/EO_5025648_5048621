import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import helper_functions as hf

if __name__ == '__main__':
    save_dir = './Figures/Pygmo/DE/'
    hf.make_ALL_folders_for_path(save_dir + 'hello')

    # Find champion of each setting configuration for DE
    champion_dict = dict()
    for setting_dir in os.listdir('./DE_settings/'):
        current_dir = './DE_settings/' + setting_dir + '/'
        final_gen_idx = len(os.listdir(current_dir)) - 1
        final_gen_filename = current_dir + f'gen_{final_gen_idx}/population.pkl'
        # Create a key describing settings used
        dict_key = f'F={setting_dir[2:5]}; CR={setting_dir[9:]}'
        with open(final_gen_filename, 'rb') as f:
            current_population = pickle.load(f)
            champion_dict[dict_key] = {'x': current_population.champion_x, 'f': current_population.champion_f}

    # Plot fitness bar chart
    fig = plt.figure(11, figsize=(4, 4))
    ax1 = fig.add_subplot(111)

    for setting in champion_dict.keys():
        ax1.bar(setting, champion_dict[setting]['f'] / (24 * 60 ** 2))

    ax1.set_ylabel('Best Fitness [days]')
    ax1.set_ylim(ax1.get_ylim()[::-1])
    ax1.grid(axis='y')
    ax1.set_xticks(ax1.get_xticks(), ax1.get_xticklabels(), rotation=90)
    plt.tight_layout()
    plt.savefig(save_dir + 'fitness_v_setting.pdf', bbox_inches='tight')

    # The best values F, CR are:
    # 0.4 0.4
    # 0.4 1.0
    # 0.6 1.0

    best_dirs = ['F_0.4_CR_0.4']

