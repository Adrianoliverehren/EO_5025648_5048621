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

propagators = {
    "cowell": propagation_setup.propagator.cowell, 
    "mee": propagation_setup.propagator.gauss_modified_equinoctial, 
    "usm_rod": propagation_setup.propagator.unified_state_model_modified_rodrigues_parameters}

sample_decision_variable_dic = {}      
sample_decision_variable_dic["dv"] = -.1  * np.array([5,2,3]) / np.linalg.norm(np.array([5,2,3]))
sample_decision_variable_dic['t_impulse'] = 31 * 24 * 60**2

benchmark_path = hf.sim_data_dir + "/integrator_analysis/benchmarks/rkf_56/dt=60"

max_time = 30


def make_log(i, results, len_data):
    def logger(evaluation):
        results.append(evaluation)
        print(f"{len(results)} / {len_data} completed, {len_data - len(results)} remaining")
        del evaluation
    return logger


def gen_benchmarks(mp_nodes=None, rerun_sims=True, path_to_save_plots=hf.root_dir + "/Figures/benchmarks"):
        
    input_lst = []
    
    step_sizes = [60, 120, 240, 480, 960, 1920]
    
    for step_size in step_sizes:
        rkf_45_benchmark_integrator_settings = {
            "type": "multistage.fixed",
            "step_size": step_size,
            "integrator_coeff_set": propagation_setup.integrator.CoefficientSets.rkf_45,
            "propagator": propagation_setup.propagator.cowell
        }
        
        rkf_56_benchmark_integrator_settings = {
            "type": "multistage.fixed",
            "step_size": step_size,
            "integrator_coeff_set": propagation_setup.integrator.CoefficientSets.rkf_56,
            "propagator": propagation_setup.propagator.cowell
        }
        
        input_lst.append([
            hf.sim_data_dir + f"/integrator_analysis/benchmarks/rkf_45/dt={int(step_size)}",
            rkf_45_benchmark_integrator_settings
        ])
        input_lst.append([
            hf.sim_data_dir + f"/integrator_analysis/benchmarks/rkf_56/dt={int(step_size)}",
            rkf_56_benchmark_integrator_settings
        ])
        
    if rerun_sims:
        investigate_integrators(input_lst, mp_nodes)
    
    analyze_benchmarks(step_sizes, path_to_save_plots)
    
def analyze_benchmarks(step_sizes, path_to_save_plots):
    
    time_in_days_lst = []
    x_error_lst = []
    y_error_lst = []
    z_error_lst = []
    pos_error_lst = []
    legend = []
    
    for step_size in step_sizes:
    
        num_states_45 = np.genfromtxt(hf.sim_data_dir + f"/integrator_analysis/benchmarks/rkf_45/dt={int(step_size)}/state_history.dat").T
        num_states_56 = np.genfromtxt(hf.sim_data_dir + f"/integrator_analysis/benchmarks/rkf_56/dt={int(step_size)}/state_history.dat").T
        
        time_in_days = num_states_45[0] / (60**2 * 24)
        
        rkf_45_pos_interpolated = interp1d(num_states_45[0], num_states_45[1:4], kind="cubic")(num_states_45[0])
        rkf_56_pos_interpolated = interp1d(num_states_56[0], num_states_56[1:4], kind="cubic")(num_states_45[0])
        
        num_state_error = rkf_45_pos_interpolated - rkf_56_pos_interpolated
        
        pos_error = np.linalg.norm(num_state_error[:3], axis=0)
              
        time_in_days_lst.append(time_in_days)  
        x_error_lst.append(num_state_error[0])
        y_error_lst.append(num_state_error[1])
        z_error_lst.append(num_state_error[2])
        pos_error_lst.append(pos_error)
        legend.append(f"dt={int(step_size)}")
        
    hf.plot_arrays(time_in_days_lst, pos_error_lst, keep_in_memory=False, x_label="Time [days]",
                y_label="Position error [m]", legend=legend, path_to_save=path_to_save_plots + "/benchmark_error_vs_time.pdf",
                y_log=True, plot_size=[4,4])

def run_sim_for_integrator_analysis(path_to_save_data, integrator_settings_dic):
    sim.run_simulation(path_to_save_data, maximum_duration=6*31*24*60**2, 
                    termination_latitude=np.deg2rad(500), termination_longitude=np.deg2rad(500), 
                    integrator_settings_dic=integrator_settings_dic, decision_variable_dic=sample_decision_variable_dic,
                    max_cpu_time=max_time)

def get_integrator_investigation_input_list(
    fixed_multistep = False,
    variable_multistep = False,
    fixed_extrapolation = False,
    variable_extrapolation = False,
    propagators_to_use = ["cowell"]
    ):
    
    fixed_stepsizes = 2.**np.arange(7, 11, 1)
    
    tolerances = 10.**np.arange(-16, -9, 2)
        
    no_steps = 2.**np.arange(1, 5, 1)
    
    input_list = []
    
    for prop_name in propagators:
        if prop_name in propagators_to_use:
            if fixed_multistep:
                # fixed multistep integrators
                for stepsize in fixed_stepsizes:
                    for key, value in RK_integrators_fixed.items():
                        path_to_save_data = hf.sim_data_dir + f"/integrator_analysis/{prop_name}/multistage/fixed/{key}/dt={int(stepsize)}"
                        integrator_settings_dic = {
                            "type": "multistage.fixed",
                            "step_size": stepsize,
                            "integrator_coeff_set": value,
                            "propagator": propagators[prop_name]
                        }
                        input_list.append([path_to_save_data, integrator_settings_dic])
            if variable_multistep:
                # variable multistep integrators
                for atol in tolerances:
                    for rtol in tolerances:
                        for key, value in RK_integrators_variable.items():
                            path_to_save_data = hf.sim_data_dir + f"/integrator_analysis/{prop_name}/multistage/variable/{key}/atol={atol:.0e}_rtol={rtol:.0e}"
                            integrator_settings_dic = {
                                "type": "multistage.variable",
                                "rel_tol": rtol,
                                "abs_tol": atol,
                                "integrator_coeff_set": value,
                                "propagator": propagators[prop_name]
                            }
                            input_list.append([path_to_save_data, integrator_settings_dic])
                            
            if fixed_extrapolation:
                # fixed extrapolation integrators
                for stepsize in fixed_stepsizes:
                    for step in no_steps:
                        for key, value in extrapolarion_integrators.items():
                            path_to_save_data = hf.sim_data_dir + f"/integrator_analysis/{prop_name}/extrapolation/fixed/{key}/max_no_steps={int(step)}/dt={int(stepsize)}"
                            integrator_settings_dic = {
                                "type": "extrapolation.fixed",
                                "extrapolation_max_no_steps": int(step),
                                "step_size": stepsize,
                                "integrator_coeff_set": value,
                                "propagator": propagators[prop_name]
                            }
                            input_list.append([path_to_save_data, integrator_settings_dic])
            if variable_extrapolation:
                # variable extrapolation integrators
                for atol in tolerances:
                    for rtol in tolerances:
                        for step in no_steps:
                            for key, value in extrapolarion_integrators.items():
                                path_to_save_data = hf.sim_data_dir + f"/integrator_analysis/{prop_name}/extrapolation/variable/{key}/max_no_steps={int(step)}/atol={atol:.0e}_rtol={rtol:.0e}"
                                integrator_settings_dic = {
                                    "type": "extrapolation.variable",
                                    "extrapolation_max_no_steps": int(step),
                                    "rel_tol": rtol,
                                    "abs_tol": atol,
                                    "integrator_coeff_set": value,
                                    "propagator": propagators[prop_name]
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
    bench_num_states_interpolator = interp1d(bench_num_states[0], bench_num_states[1:], kind="cubic")
    bench_end_t = bench_num_states[0][-1]
    
    input_list = []
    for inp in old_input_list:
        input_list.append([inp[0], bench_num_states_interpolator, bench_end_t])
    

    if mp_nodes:    
        pool = mp.Pool(mp_nodes, maxtasksperchild=4)
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

def remove_path_prefix(path, prefix):
    if path.startswith(prefix):
        return os.path.relpath(path, prefix)
    else:
        return path

def compare_integrator_to_benchmark(folder_path, bench_num_states_interpolator, bench_end_t):
    
    # plot_title = remove_path_prefix(folder_path, hf.sim_data_dir + "/integrator_analysis")
    # plot_path = plot_title.replace("\\", "_")
    
    # print(plot_path)
    
    # sys.exit()
    if os.path.exists(folder_path):
        
        num_states = np.genfromtxt(folder_path + "/state_history.dat").T
        
        end_t = num_states[0][-1]
        
        integrator_eval_dict = {}
        if end_t * 0.995 < bench_end_t < end_t * 1.005:
            # propagation ended close to the required point
            integrator_eval_dict["finalized_correctly"] = True
            
            shortest_time = min(bench_end_t, end_t)
            eval_array = np.linspace(0, shortest_time, 2000)
            
            eval_array_days = eval_array / (24*60**2)
            
            num_states_interpolatored = interp1d(num_states[0], num_states[1:], kind="cubic")(eval_array)
            bench_num_states_interpolated = bench_num_states_interpolator(eval_array)
            
            x_bench, y_bench, z_bench = bench_num_states_interpolated[0], bench_num_states_interpolated[1], bench_num_states_interpolated[2]
            x, y, z = num_states_interpolatored[0], num_states_interpolatored[1], num_states_interpolatored[2]
            
            error_states = bench_num_states_interpolated - num_states_interpolatored
            
            save_error_states = np.zeros((len(error_states.T[0])+1, len(error_states[0])))
            
            save_error_states[0] = eval_array
            save_error_states[1:] = error_states
            
            np.savetxt(folder_path + "/state_errors.dat", save_error_states.T)        
            
            pos_error = np.linalg.norm(error_states[0:3], axis=0)
            
            plot_title = remove_path_prefix(folder_path, hf.sim_data_dir + "/integrator_analysis")
            plot_path = plot_title.replace("\\", "_")
            
            hf.plot_arrays(
                eval_array_days, [pos_error], x_label="Time [days]", y_labels="Error [m]", 
                title=plot_title, path_to_save=hf.root_dir + f"/Figures/integrator_analysis/errors_over_time/{plot_path}.pdf",
                y_log=True, plot_size=[6,6])
            
            hf.plot_arrays(
                eval_array_days, [x_bench, y_bench, z_bench, x, y, z], x_label="Time [days]", y_labels="Pos [m]", 
                title=plot_title, path_to_save=hf.root_dir + f"/Figures/integrator_analysis/position_over_time/{plot_path}.pdf",
                plot_size=[6,6], legend=["x_bench", "y_bench", "z_bench", "x", "y", "z"])
        
            integrator_eval_dict["max_pos_error"] = np.max(pos_error)        
            
        else:
            integrator_eval_dict["finalized_correctly"] = False
            
            
        hf.save_dict_to_json(integrator_eval_dict, folder_path + "/integrator_eval_dict.dat")
        
def create_integrator_analysis_plots(input_lst, path_to_save_plots, regen_data=True, show_all_integrators=False):
        
    scatter_plot_data = [[],[]]
    
    label_points = []
    
    if regen_data:
        k = 0
        for inp in input_lst:
            if os.path.exists(inp[0] + "/integrator_eval_dict.dat"):
                integrator_eval_dict = hf.create_dic_drom_json(inp[0] + "/integrator_eval_dict.dat")
                
                if integrator_eval_dict["finalized_correctly"]:
                    propagation_info_dic = hf.create_dic_drom_json(inp[0] + "/propagation_info_dic.dat")
                    
                    if integrator_eval_dict["max_pos_error"] < 1e6 and propagation_info_dic["f_evals"] < 500000 and integrator_eval_dict["max_pos_error"] > 10:                
                        scatter_plot_data[1].append(integrator_eval_dict["max_pos_error"])
                        scatter_plot_data[0].append(propagation_info_dic["f_evals"])
                        
                        integrator_name = remove_path_prefix(inp[0], hf.sim_data_dir + "/integrator_analysis")
                        
                        if integrator_name == "cowell\\multistage\\variable\\rkdp_87\\atol=1e-12_rtol=1e-10":
                            
                            selected_id = k
                        
                        k+=1
                        label_points.append({
                            "text": integrator_name,
                            "x": propagation_info_dic["f_evals"],
                            "y": integrator_eval_dict["max_pos_error"],
                            "x_text": 10,
                            "y_text": 10,
                        })
        np.save(hf.sim_data_dir + "/integrator_analysis/plotting_data/scatter_plot_data.npy", scatter_plot_data) 
        np.save(hf.sim_data_dir + "/integrator_analysis/plotting_data/label_points.npy", label_points) 
        
    else:
        scatter_plot_data = np.load(hf.sim_data_dir + "/integrator_analysis/plotting_data/scatter_plot_data.npy", allow_pickle=True)
        label_points = np.load(hf.sim_data_dir + "/integrator_analysis/plotting_data/label_points.npy", allow_pickle=True)
    
    
    plt.figure(figsize=[4,4])
    
    plt.scatter(scatter_plot_data[0], scatter_plot_data[1], marker=".")
    plt.scatter(scatter_plot_data[0][selected_id], scatter_plot_data[1][selected_id], color="tab:red", marker="o")
    
    plt.xlabel("Function evaluations [-]")
    plt.ylabel("Position error [m]")
    plt.yscale("log")
    
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    
    plt.grid()
    
    plt.tight_layout()
    
    plt.savefig(path_to_save_plots + "/integrators_error_vs_fevals.pdf", bbox_inches='tight')
    
    plt.close()
    
    if show_all_integrators:
        hf.plot_arrays(scatter_plot_data[0], [scatter_plot_data[1]], linewiths=[0]*len(scatter_plot_data[0]),
                    y_log=True, y_label="Pos error [m]", x_label="f-evals [-]", 
                    label_points=label_points, keep_in_memory=True, markings=True)
        plt.show()   

def plot_error_for_single_integrator(integrator_path, plot_save_path, plot_name):
    
    error_states = np.genfromtxt(integrator_path + "/state_errors.dat").T   
    num_states = np.genfromtxt(integrator_path + "/state_history.dat").T   
    
    eval_array_days = error_states[0] / (24*60**2)
    
    pos_error = np.linalg.norm(error_states[1:4], axis=0)
    
    hf.plot_arrays(
        eval_array_days, [pos_error], x_label="Time [days]", y_label="Position error [m]", 
        path_to_save=plot_save_path + f"/{plot_name}_error_vs_tome.pdf",
        y_log=True, plot_size=[4,4])
    
    hf.plot_arrays(
        num_states[0]/ (24*60**2), [np.diff(num_states[0], append=np.nan)], x_label="Time [days]", y_label="Timestep [s]", 
        path_to_save=plot_save_path + f"/{plot_name}_dt_vs_tome.pdf",
        plot_size=[4,4])    
    
    pass


if __name__ == "__main__":
        
    # print(hf.report_dir )
        
    # gen_benchmarks(mp_nodes=15, rerun_sims=False, path_to_save_plots=hf.report_dir + "/Figures/Ch2")
        
    # sys.exit()    
    
    # print(hf.root_dir)
    
    # input_lst = get_integrator_investigation_input_list(True)
    # input_lst = get_integrator_investigation_input_list(True, True, False, False)
    # input_lst = get_integrator_investigation_input_list(True, True, True, True, ["cowell", "mee", "usm_rod"])
    # input_lst = get_integrator_investigation_input_list(True, True, True, True, ["cowell"])
    
    # print(len(input_lst))
        
    # investigate_integrators(input_lst, 15)
    
    # compare_integrators_with_mp(input_lst, 16) 
    
    # input_lst = get_integrator_investigation_input_list(True, True, True, True, ["cowell", "mee", "usm_rod"])
    
    # create_integrator_analysis_plots(input_lst, hf.report_dir + "/Figures/Ch2", regen_data=True, show_all_integrators=True)
    
    plot_error_for_single_integrator(hf.sim_data_dir + "/integrator_analysis/cowell\\multistage\\variable\\rkdp_87\\atol=1e-12_rtol=1e-10", 
                                     hf.report_dir + "/Figures/Ch2",
                                     plot_name="cowell_rkdp_87_atol=1e-12_rtol=1e-10")
    plot_error_for_single_integrator(hf.sim_data_dir + "/integrator_analysis/mee\\multistage\\variable\\rkf_45\\atol=1e-12_rtol=1e-12", 
                                     hf.report_dir + "/Figures/Ch2",
                                     plot_name="mee_rkf_45_atol=1e-12_rtol=1e-12")
        
    # analyze_benchmarks()
    
    # gen_benchmarks()
    