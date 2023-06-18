import numpy as np
import simulation as sim
from tudatpy.kernel.numerical_simulation import propagation_setup
import helper_functions as hf

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
sample_decision_variable_dic['t_impulse'] = 31 * 24* 60**2


def gen_benchmarks():
    
    rkf_45_benchmark_integrator_settings = {
    "type": "multistage.fixed",
    "step_size": 10,
    "integrator_coeff_set": propagation_setup.integrator.CoefficientSets.rkf_45
    }
    
    sim.run_simulation(hf.sim_data_dir + "/integrator_analysis/benchmarks/rkf_45", maximum_duration=6*31*24*60**2, 
                       termination_latitude=np.deg2rad(90), termination_longitude=np.deg2rad(90), 
                       integrator_settings_dic=rkf_45_benchmark_integrator_settings)
    
    print("DONE with rkf45")
    
    rkf_78_benchmark_integrator_settings = {
    "type": "multistage.fixed",
    "step_size": 10,
    "integrator_coeff_set": propagation_setup.integrator.CoefficientSets.rkf_78
    }
    
    sim.run_simulation(hf.sim_data_dir + "/integrator_analysis/benchmarks/rkf_78", maximum_duration=6*31*24*60**2, 
                       termination_latitude=np.deg2rad(90), termination_longitude=np.deg2rad(90), 
                       integrator_settings_dic=rkf_78_benchmark_integrator_settings)
    
    print("DONE with rkf78")
    
if __name__ == "__main__":
    
    print(hf.sim_data_dir)
    
    gen_benchmarks()
    
    
    












