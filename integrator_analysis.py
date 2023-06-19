import numpy as np
import simulation as sim
from tudatpy.kernel.numerical_simulation import propagation_setup
import helper_functions as hf
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import multiprocessing as mp
import os
import sys

RK_integrators_fixed = {
    "euler_forward": propagation_setup.integrator.CoefficientSets.euler_forward,   
    "rk_4": propagation_setup.integrator.CoefficientSets.rk_4,
    "explicit_trapezoid_rule": propagation_setup.integrator.CoefficientSets.explicit_trapezoid_rule,
    "three_eight_rule_rk_4": propagation_setup.integrator.CoefficientSets.three_eight_rule_rk_4,
    "rkf_45": propagation_setup.integrator.CoefficientSets.rkf_45,
    "rkf_56": propagation_setup.integrator.CoefficientSets.rkf_56,
    "rkf_78": propagation_setup.integrator.CoefficientSets.rkf_78,
    "rkdp_87": propagation_setup.integrator.CoefficientSets.rkdp_87,
    "rkf_1210": propagation_setup.integrator.CoefficientSets.rkf_1210,
    "rkf_1412": propagation_setup.integrator.CoefficientSets.rkf_1412}

RK_integrators_fixed_nice_names = [
    "Euler forward",
    "RK 4",
    "Explicit trapezoid rule",
    "Three eight rule RK4",
    "RKF 4(5)",
    "RKF 5(6)",
    "RKF 7(8)",
    "RKDP 8(7)",
    "RKF 12(10)",
    "RKF 14(12)",
]

RK_integrators_variable = {
    "rkf_45": propagation_setup.integrator.CoefficientSets.rkf_45,   
    "rkf_56": propagation_setup.integrator.CoefficientSets.rkf_56,
    "rkf_78": propagation_setup.integrator.CoefficientSets.rkf_78,
    "rkdp_87": propagation_setup.integrator.CoefficientSets.rkdp_87,
    "rkf_1210": propagation_setup.integrator.CoefficientSets.rkf_1210,
    "rkf_1412": propagation_setup.integrator.CoefficientSets.rkf_1412
}

extrapolarion_integrators = {
    "bulirsch_stoer_sequence": propagation_setup.integrator.ExtrapolationMethodStepSequences.bulirsch_stoer_sequence,
    "deufelhard_sequence": propagation_setup.integrator.ExtrapolationMethodStepSequences.deufelhard_sequence
}

sample_decision_variable_dic = {}
sample_decision_variable_dic["dv_mag"] = -.1        
sample_decision_variable_dic["dv_unit_vect"] = np.array([5,2,3]) / np.linalg.norm(np.array([5,2,3]))
sample_decision_variable_dic['t_impulse'] = 31 * 24 * 60**2


benchmark_path = hf.sim_data_dir + "/integrator_analysis/benchmarks/rkf_78"

def make_log(i, results, len_data):
    def logger(evaluation):
        results.append(evaluation)
        print(f"{len(results)} / {len_data} completed, {len_data - len(results)} remaining")
        del evaluation
    return logger

def gen_benchmarks():
    
    print("Starting integrator analysis")
    
    rkf_45_benchmark_integrator_settings = {
    "type": "multistage.fixed",
    "step_size": 60,
    "integrator_coeff_set": propagation_setup.integrator.CoefficientSets.rkf_45
    }
    
    sim.run_simulation(hf.sim_data_dir + "/integrator_analysis/benchmarks/rkf_45", maximum_duration=6*31*24*60**2, 
                       termination_latitude=np.deg2rad(500), termination_longitude=np.deg2rad(500), 
                       integrator_settings_dic=rkf_45_benchmark_integrator_settings, decision_variable_dic=sample_decision_variable_dic,
                       max_cpu_time=30*60)
    
    print("DONE with rkf45")
    
    rkf_78_benchmark_integrator_settings = {
    "type": "multistage.fixed",
    "step_size": 60,
    "integrator_coeff_set": propagation_setup.integrator.CoefficientSets.rkf_78
    }
    
    sim.run_simulation(hf.sim_data_dir + "/integrator_analysis/benchmarks/rkf_78", maximum_duration=6*31*24*60**2, 
                       termination_latitude=np.deg2rad(500), termination_longitude=np.deg2rad(500), 
                       integrator_settings_dic=rkf_78_benchmark_integrator_settings, decision_variable_dic=sample_decision_variable_dic,
                       max_cpu_time=30*60)
    
    print("DONE with rkf78")

def plot_benchmarks():
    
    num_states_45 = np.genfromtxt(hf.sim_data_dir + "/integrator_analysis/benchmarks/rkf_45/state_history.dat").T
    num_states_78 = np.genfromtxt(hf.sim_data_dir + "/integrator_analysis/benchmarks/rkf_78/state_history.dat").T
    
    time_in_days = num_states_45[0] / (60**2 * 24)
    
    rkf_45_pos_interpolated = interp1d(num_states_45[0], num_states_45[1:4])(num_states_45[0])
    rkf_78_pos_interpolated = interp1d(num_states_78[0], num_states_78[1:4])(num_states_45[0])
    
    x_error = rkf_45_pos_interpolated[0] - rkf_78_pos_interpolated[0]
    y_error = rkf_45_pos_interpolated[1] - rkf_78_pos_interpolated[1]
    z_error = rkf_45_pos_interpolated[2] - rkf_78_pos_interpolated[2]
    
    hf.plot_arrays(time_in_days, [x_error, y_error, z_error], keep_in_memory=True, x_label="Time [days]",
                   y_label="pos error [m]", legend=["x", "y", "z"])

def run_sim_for_integrator_analysis(path_to_save_data, integrator_settings_dic):
    try:
        sim.run_simulation(path_to_save_data, maximum_duration=6*31*24*60**2, 
                        termination_latitude=np.deg2rad(500), termination_longitude=np.deg2rad(500), 
                        integrator_settings_dic=integrator_settings_dic, decision_variable_dic=sample_decision_variable_dic,
                        max_cpu_time=5)
    except:
        pass

def get_integrator_investigation_input_list(
    fixed_multistep = False,
    variable_multistep = False,
    fixed_extrapolation = False,
    variable_extrapolation = False,
    ):
    
    fixed_stepsizes = 2.**np.arange(7, 13, 1)
    
    tolerances = 10.**np.arange(-12, -3, 1)
    
    no_steps = 2.**np.arange(1, 8, 1)
    
    input_list = []
    
    if fixed_multistep:
        # fixed multistep integrators
        for stepsize in fixed_stepsizes:
            for key, value in RK_integrators_fixed.items():
                path_to_save_data = hf.sim_data_dir + f"/integrator_analysis/multistage/fixed/{key}/dt={int(stepsize)}"
                integrator_settings_dic = {
                    "type": "multistage.fixed",
                    "step_size": stepsize,
                    "integrator_coeff_set": value
                }
                input_list.append([path_to_save_data, integrator_settings_dic])
    if variable_multistep:
        # variable multistep integrators
        for atol in tolerances:
            for rtol in tolerances:
                for key, value in RK_integrators_variable.items():
                    path_to_save_data = hf.sim_data_dir + f"/integrator_analysis/multistage/variable/{key}/atol={atol:.0e}_rtol={rtol:.0e}"
                    integrator_settings_dic = {
                        "type": "multistage.variable",
                        "rel_tol": rtol,
                        "abs_tol": atol,
                        "integrator_coeff_set": value
                    }
                    input_list.append([path_to_save_data, integrator_settings_dic])
                    
    if fixed_extrapolation:
        # fixed extrapolation integrators
        for stepsize in fixed_stepsizes:
            for step in no_steps:
                for key, value in extrapolarion_integrators.items():
                    path_to_save_data = hf.sim_data_dir + f"/integrator_analysis/extrapolation/fixed/{key}/max_no_steps={int(step)}/dt={int(stepsize)}"
                    integrator_settings_dic = {
                        "type": "extrapolation.fixed",
                        "extrapolation_max_no_steps": int(step),
                        "step_size": stepsize,
                        "integrator_coeff_set": value
                    }
                    input_list.append([path_to_save_data, integrator_settings_dic])
    if variable_extrapolation:
        # variable extrapolation integrators
        for atol in tolerances:
            for rtol in tolerances:
                for step in no_steps:
                    for key, value in extrapolarion_integrators.items():
                        path_to_save_data = hf.sim_data_dir + f"/integrator_analysis/extrapolation/variable/{key}/max_no_steps={int(step)}/atol={atol:.0e}_rtol={rtol:.0e}"
                        integrator_settings_dic = {
                            "type": "extrapolation.variable",
                            "extrapolation_max_no_steps": int(step),
                            "rel_tol": rtol,
                            "abs_tol": atol,
                            "integrator_coeff_set": value
                        }
                        input_list.append([path_to_save_data, integrator_settings_dic])
                        
    return input_list

def investigate_integrators(
    input_list,
    mp_nodes=None
):    

    if mp_nodes:    
        pool = mp.Pool(mp_nodes, maxtasksperchild=40)
        results = []
        for i, input in enumerate(input_list):
            pool.apply_async(run_sim_for_integrator_analysis, args=input, callback=make_log(i, results, len(input_list)))
        pool.close()
        pool.join()  
    
    else:
        i = 0
        for input in input_list:
            _ = run_sim_for_integrator_analysis(*input)
            i += 1
            print(f"{i} / {len(input_list)} completed")
            

def compare_integrators_with_mp(
    old_input_list,
    mp_nodes=None
):    
    
    bench_num_states = np.genfromtxt(benchmark_path + "/state_history.dat").T
    bench_num_states_interpolator = interp1d(bench_num_states[0], bench_num_states[1:])
    bench_end_t = bench_num_states[0][-1]
    
    input_list = []
    for inp in old_input_list:
        input_list.append([inp[0], bench_num_states_interpolator, bench_end_t])
    

    if mp_nodes:    
        pool = mp.Pool(mp_nodes, maxtasksperchild=40000)
        results = []
        for i, input in enumerate(input_list):
            pool.apply_async(compare_integrator_to_benchmark, args=input, callback=make_log(i, results, len(input_list)))
        pool.close()
        pool.join()  
    
    else:
        i = 0
        for input in input_list:
            _ = compare_integrator_to_benchmark(*input)
            i += 1
            print(f"{i} / {len(input_list)} completed")

def compare_integrator_to_benchmark(folder_path, bench_num_states_interpolator, bench_end_t):
    
    if os.path.exists(folder_path):
        
        num_states = np.genfromtxt(folder_path + "/state_history.dat").T
        
        end_t = num_states[0][-1]
        
        integrator_eval_dict = {}
        
        if end_t * 0.995 < bench_end_t < end_t * 1.005:
            # propagation ended close to the required point
            integrator_eval_dict["finalized_correctly"] = True
            
            shortest_time = min(bench_end_t, end_t)
            eval_array = np.arange(0, shortest_time+60, 60)
            
            num_states_interpolatored = interp1d(num_states[0], num_states[1:])(eval_array)
            bench_num_states_interpolated = bench_num_states_interpolator(eval_array)
            
            error_states = bench_num_states_interpolated - num_states_interpolatored
            
            save_error_states = np.zeros((len(error_states.T[0])+1, len(error_states[0])))
            
            save_error_states[0] = eval_array
            save_error_states[1:] = error_states
            
            np.savetxt(folder_path + "/state_errors.dat", save_error_states.T)        
            
            pos_error = np.linalg.norm(error_states[0:3], axis=0)
        
            integrator_eval_dict["max_pos_error"] = np.max(pos_error)        
            
        else:
            integrator_eval_dict["finalized_correctly"] = False
            
            
        hf.save_dict_to_json(integrator_eval_dict, folder_path + "/integrator_eval_dict.dat")
        
def create_integrator_analysis_plots():
    
    
    all_integratos_input_lst = get_integrator_investigation_input_list(True, True, True, True)
    
    scatter_plot_data = [[],[]]
    
    for inp in all_integratos_input_lst:
        if os.path.exists(inp[0]):
            integrator_eval_dict = hf.create_dic_drom_json(inp[0] + "/integrator_eval_dict.dat")
            
            if integrator_eval_dict["finalized_correctly"]:
                
                scatter_plot_data[1].append(integrator_eval_dict["max_pos_error"])
                propagation_info_dic = hf.create_dic_drom_json(inp[0] + "/propagation_info_dic.dat")
                scatter_plot_data[0].append(propagation_info_dic["f_evals"])
                
    plt.figure()
    
    plt.scatter(scatter_plot_data[0], scatter_plot_data[1])
    
    plt.yscale("log")
                
    plt.show()
        
        



if __name__ == "__main__":
        
    print(hf.root_dir)
    
    # input_lst = get_integrator_investigation_input_list(True)
    # input_lst = get_integrator_investigation_input_list(True, True, False, False)
    # input_lst = get_integrator_investigation_input_list(True, True, False, False)
        
    # investigate_integrators(input_lst, 15)
    
    # compare_integrators_with_mp(input_lst, 15)
    
    create_integrator_analysis_plots()
    
        
    # plot_benchmarks()
    
    
    
    # gen_benchmarks()
    
    
    












