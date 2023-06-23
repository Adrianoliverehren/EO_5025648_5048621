# General imports
import os
import pdb

import numpy as np
import matplotlib.pyplot as plt
import sys
import json
import gc
import pathlib
import helper_functions as hf
import simulation_setup_functions as ssf


# Tudatpy imports
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.astro import element_conversion, frame_conversion
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import environment
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.math import interpolators


spice_interface.load_standard_kernels()

bodies_to_create = ["Earth", "Moon", "Sun"]

default_init_mee = np.zeros(6)
default_init_mee[0] = 42164177
default_init_mee[-1] = np.deg2rad(-80+0.147)

default_integrator_settings_dic = {
        "type": "multistage.variable",
        "rel_tol": 1e-10,
        "abs_tol": 1e-12,
        "integrator_coeff_set": propagation_setup.integrator.CoefficientSets.rkdp_87,
        "propagator": propagation_setup.propagator.cowell
    }

default_dep_vars = [
        propagation_setup.dependent_variable.altitude("SUPER_SAT_37k", "Earth"),
        propagation_setup.dependent_variable.body_fixed_groundspeed_velocity("SUPER_SAT_37k", "Earth"), 
        propagation_setup.dependent_variable.geodetic_latitude("SUPER_SAT_37k", "Earth"),
        propagation_setup.dependent_variable.latitude("SUPER_SAT_37k", "Earth"),
        propagation_setup.dependent_variable.longitude("SUPER_SAT_37k", "Earth"),
        propagation_setup.dependent_variable.total_acceleration("SUPER_SAT_37k"),
        # propagation_setup.dependent_variable.single_acceleration(
        #     acceleration_type=propagation_setup.acceleration.point_mass_gravity_type,
        #     body_undergoing_acceleration="SUPER_SAT_37k",
        #     body_exerting_acceleration="Earth"),
        propagation_setup.dependent_variable.single_acceleration(
            acceleration_type=propagation_setup.acceleration.point_mass_gravity_type,
            body_undergoing_acceleration="SUPER_SAT_37k",
            body_exerting_acceleration="Sun"),
        propagation_setup.dependent_variable.single_acceleration(
            acceleration_type=propagation_setup.acceleration.point_mass_gravity_type,
            body_undergoing_acceleration="SUPER_SAT_37k",
            body_exerting_acceleration="Moon"),
]
default_decision_variable_dic = {'dv': np.array([0, 0, 0]), 't_impulse': 0.5*24*60**2}


def get_empty_body_settings(
    global_frame_origin = 'Earth',
    global_frame_orientation = 'J2000'
):

    # Define settings for celestial bodies
    bodies_to_create = ["Earth", "Moon", "Sun"]
    
    empty_body_settings = environment_setup.BodyListSettings(
        frame_origin = global_frame_origin, 
        frame_orientation = global_frame_orientation)

    # Create empty body settings
    
    for body in bodies_to_create:
        empty_body_settings.add_empty_settings(body)
    
    return empty_body_settings


def run_simulation(
    path_to_save_data,
    maximum_duration,
    decision_variable_dic=default_decision_variable_dic,
    simulation_start_epoch=0,
    termination_latitude=hf.constraint() * 1.5,
    termination_longitude=hf.constraint() * 1.5,
    integrator_settings_dic=default_integrator_settings_dic,
    max_cpu_time=30,
    sim_idx=0,
    run_for_mc=True,
):
    empty_body_settings = get_empty_body_settings()
    
    body_settings = environment_setup.get_default_body_settings(
        bodies=bodies_to_create, 
        base_frame_origin=empty_body_settings.frame_origin, 
        base_frame_orientation=empty_body_settings.frame_orientation
    )
        
    bodies = environment_setup.create_system_of_bodies(body_settings)
    bodies.create_empty_body("SUPER_SAT_37k")

    ######################################################################################

    # TERMINATION CONDITIONS

    ######################################################################################

    # Check if V impulse at t!=0
    if decision_variable_dic:
        time_termination_settings = propagation_setup.propagator.time_termination(
            simulation_start_epoch + decision_variable_dic['t_impulse'],
            terminate_exactly_on_final_condition=True
        )
    else:
        time_termination_settings = propagation_setup.propagator.time_termination(
            simulation_start_epoch + maximum_duration,
            terminate_exactly_on_final_condition=True
        )

    general_termination_settings = ssf.get_general_termination_settings(termination_latitude, termination_longitude,
                                                                        max_cpu_time)

    hybrid_termination_conditions_list = [time_termination_settings] + general_termination_settings

    # Create termination settings object (when either the time of altitude condition is reached: propaation terminates)
    hybrid_termination_settings = propagation_setup.propagator.hybrid_termination(
        hybrid_termination_conditions_list,
        fulfill_single_condition=True
        )   

    # Get acceleration, integrator and propagator settings
    acceleration_models = ssf.get_acceleration_settings(bodies)
    integrator_settings = ssf.get_integrator_settings(integrator_settings_dic)
    
    mu = bodies.get("Earth").gravitational_parameter
    
    cartesian_init_state = element_conversion.mee_to_cartesian(
        default_init_mee,
        mu,
        False
    )
    
    if decision_variable_dic['t_impulse'] == 0:
        rsw_delta_v = decision_variable_dic["dv"] * decision_variable_dic["dv_unit_vect"]
        # Rotate delta_v to cartesian frame
        rotation_matrix = frame_conversion.inertial_to_rsw_rotation_matrix(cartesian_init_state)
        delta_v_cartesian = rotation_matrix @ rsw_delta_v

        cartesian_init_state[3:6] += delta_v_cartesian
    
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies=["Earth"],
        acceleration_models=acceleration_models,
        bodies_to_integrate=["SUPER_SAT_37k"],
        initial_states=cartesian_init_state,
        initial_time=simulation_start_epoch,
        integrator_settings=integrator_settings,
        termination_settings=hybrid_termination_settings,
        propagator=integrator_settings_dic["propagator"],
        output_variables=default_dep_vars)
    
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies, propagator_settings)
        
    first_cpu_time = list(dynamics_simulator.cumulative_computation_time_history.values())[-1]
    first_f_evals = list(dynamics_simulator.cumulative_number_of_function_evaluations.values())[-1]

    # If terminated based on reaching impulse time
    if list(dynamics_simulator.state_history.keys())[-1] == decision_variable_dic['t_impulse']:
        # Apply velocity impulse
        current_epoch = list(dynamics_simulator.state_history.keys())[-1]
        current_state = dynamics_simulator.state_history[current_epoch].copy()
        rsw_delta_v = decision_variable_dic["dv"]

        # Rotate delta_v to cartesian frame
        rotation_matrix = frame_conversion.inertial_to_rsw_rotation_matrix(current_state)
        delta_v_cartesian = rotation_matrix @ rsw_delta_v
        current_state[3:6] += delta_v_cartesian

        # Update termination settings
        time_termination_settings = propagation_setup.propagator.time_termination(
            current_epoch + maximum_duration,
            terminate_exactly_on_final_condition=True
        )

        new_termination_settings_list = [time_termination_settings] + general_termination_settings
        hybrid_termination_settings = propagation_setup.propagator.hybrid_termination(
            new_termination_settings_list,
            fulfill_single_condition=True
        )

        propagator_settings = propagation_setup.propagator.translational(
                                    central_bodies=["Earth"],
                                    acceleration_models=acceleration_models,
                                    bodies_to_integrate=["SUPER_SAT_37k"],
                                    initial_states=current_state,
                                    initial_time=current_epoch,
                                    integrator_settings=integrator_settings,
                                    termination_settings= hybrid_termination_settings,
                                    propagator=propagation_setup.propagator.cowell,
                                    output_variables=default_dep_vars)

        dynamics_simulator_2 = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)
        second_cpu_time = list(dynamics_simulator.cumulative_computation_time_history.values())[-1]
        second_f_evals = list(dynamics_simulator.cumulative_number_of_function_evaluations.values())[-1]
        
    else:
        dynamics_simulator_2 = None
        second_cpu_time = 0
        second_f_evals = 0
    
    dep_vars_id_dic = hf.make_dic_safe_for_json(dynamics_simulator.propagation_results.dependent_variable_ids)
    safe_decision_variable_dic = hf.make_dic_safe_for_json(decision_variable_dic)
    
    integrator_settings_dic_for_save = integrator_settings_dic.copy()
    
    integrator_settings_dic_for_save["integrator_coeff_set"] = str(integrator_settings_dic_for_save["integrator_coeff_set"])
    integrator_settings_dic_for_save["propagator"] = str(integrator_settings_dic_for_save["propagator"])
    
    propagation_info_dic = {
        "CPU_time": first_cpu_time + second_cpu_time,
        "f_evals": first_f_evals + second_f_evals,
        "decision_variable_dic": safe_decision_variable_dic,
        "dependent_variable_ids": dep_vars_id_dic,
        "integrator_info": integrator_settings_dic_for_save
    }        
    if dynamics_simulator_2 is None:
        stacked_state_history = dynamics_simulator.state_history
        stacked_dep_vars_history = dynamics_simulator.dependent_variable_history
    else:
        stacked_state_history = dynamics_simulator.state_history | dynamics_simulator_2.state_history
        stacked_dep_vars_history = dynamics_simulator.dependent_variable_history | \
            dynamics_simulator_2.dependent_variable_history

    if path_to_save_data:

        hf.save_dict_to_json(propagation_info_dic, path_to_save_data + "/propagation_info_dic.dat")
        hf.save_dynamics_simulator_to_files(path_to_save_data, stacked_state_history, stacked_dep_vars_history)
    
    if run_for_mc:
        
        return sim_idx, [hf.calculate_obj(stacked_dep_vars_history, sim_idx),
                        hf.period_change(stacked_state_history, decision_variable_dic['t_impulse'], stacked_dep_vars_history)]
        


if __name__ == "__main__":
    
    decision_variable_dic = {}
    
    decision_variable_dic["dv_mag"] = -.1        
    decision_variable_dic["dv_unit_vect"] = np.array([0, 1, 0])
    decision_variable_dic['t_impulse'] = 0.5 * 24 * 60**2
    print(decision_variable_dic)
    run_simulation("test2", 2 * 31 * 24 * 60**2)


