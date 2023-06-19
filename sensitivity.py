import pdb
import matplotlib.pyplot as plt

import numpy as np
import os
import helper_functions as hf


def log_sensitivity(x, f, dxdf):
    """
    Finds the log sensitivity as a function of decision variable x
    """
    return (x/f) * dxdf


if __name__ == '__main__':
    # Read objective values from Monte Carlo
    t_max_arr = np.genfromtxt('./DesignSpace/monte_carlo/objectives_constraints.dat')[:, 1]
    # Transform to days
    t_max_arr = t_max_arr / (24 * 60 * 60)

    # Read decision variable values from Monte Carlo
    v1_arr = []
    v2_arr = []
    v3_arr = []
    t_impulse_arr = []
    #
    for i in range(128):
        sim_dir = f'sim{i}'
        full_dir = './DesignSpace/monte_carlo/' + sim_dir + '/'
        if sim_dir != 'objectives_constraints.dat' and sim_dir != 'parameter_values.dat':
            decision_var_dict = hf.create_dic_drom_json(full_dir + 'propagation_info_dic.dat')['decision_variable_dic']

            v1_arr.append(decision_var_dict['dv_mag'] * decision_var_dict['dv_unit_vect'][0])
            v2_arr.append(decision_var_dict['dv_mag'] * decision_var_dict['dv_unit_vect'][1])
            v3_arr.append(decision_var_dict['dv_mag'] * decision_var_dict['dv_unit_vect'][2])
            t_impulse_arr.append(decision_var_dict['t_impulse'])
    # Transform time to days
    print(t_impulse_arr)
    t_impulse_arr = np.array(t_impulse_arr) / (24 * 60 * 60)
    print(t_impulse_arr)

    # Sort arrays such that they each are in ascending order of decision variable
    # [x for _, x in sorted(zip(Y, X))]
    # Each array should be 2 rows, row 0 is sorted decision, row 1 = corresponding sorted objective
    v1_sorted = [sorted(v1_arr), [t for _, t in sorted(zip(v1_arr, t_max_arr))]]
    v2_sorted = [sorted(v2_arr), [t for _, t in sorted(zip(v2_arr, t_max_arr))]]
    v3_sorted = [sorted(v3_arr), [t for _, t in sorted(zip(v3_arr, t_max_arr))]]
    t_impulse_sorted = [sorted(t_impulse_arr), [t for _, t in sorted(zip(t_impulse_arr, t_max_arr))]]

    print(t_impulse_sorted[0])

    # For every decision variable, finite difference
    sorted_arrays = [v1_sorted, v2_sorted, v3_sorted, t_impulse_sorted]
    log_sensitivity_matrix = []
    for sorted_array in sorted_arrays:
        log_sensitivity_row = []
        derivative_arr = np.gradient(sorted_array[1], sorted_array[0])

        # evaluate log sensitivity
        for x, f, dxdf in zip(sorted_array[0], sorted_array[1], derivative_arr):
            log_sensitivity_row.append(log_sensitivity(x, f, dxdf))
        log_sensitivity_matrix.append(log_sensitivity_row)

    # Plot
    fig = plt.figure(figsize=(4, 4))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    ax1.plot(v1_sorted[0], log_sensitivity_matrix[0], label=r'$\Delta V$ impulse in radial direction')
    ax1.plot(v2_sorted[0], log_sensitivity_matrix[1], label=r'$\Delta V$ impulse in along-track direction')
    ax1.plot(v3_sorted[0], log_sensitivity_matrix[2], label=r'$\Delta V$ impulse in cross-track direction')

    ax1.set_xlabel('Velocity Impulse [m/s]')
    ax1.set_ylabel('Log-sensitivity [-]')

    ax2.plot(t_impulse_sorted[0], log_sensitivity_matrix[3], label=r'Time of impulse', c='magenta')
    ax2.set_xlabel('Time of Impulse [days]')
    ax1.legend()
    ax2.legend()

    plt.grid()
    plt.tight_layout()

    plt.savefig('./Figures/LogSensitivity.pdf')

    plt.clf()
    plt.cla()
