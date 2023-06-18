# General imports
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import json
import gc
import pathlib
import helper_functions as hf

# Tudatpy imports
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.astro import element_conversion
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
    decision_variable_dic=None,
    simulation_start_epoch=0,
    termination_latitude=np.deg2rad(0.32),
    termination_longitude=np.deg2rad(0.32),
    integrator_stepsize=600,
    integrator_coeff_set=propagation_setup.integrator.CoefficientSets.rkf_45,
):
    
    empty_body_settings = get_empty_body_settings()
    
    body_settings = environment_setup.get_default_body_settings(
        bodies=bodies_to_create, 
        base_frame_origin=empty_body_settings.frame_origin, 
        base_frame_orientation=empty_body_settings.frame_orientation
    )
        
    bodies = environment_setup.create_system_of_bodies(body_settings)
    
    bodies.create_empty_body("SUPER_SAT_37k")
    
    time_termination_settings = propagation_setup.propagator.time_termination(
        simulation_start_epoch + maximum_duration,
        terminate_exactly_on_final_condition=False
    )
    
    ######################################################################################
    
    # TERMINATION CONDITIONS
    
    ######################################################################################
    upper_lat_termination_settings = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=propagation_setup.dependent_variable.latitude("SUPER_SAT_37k", "Earth"),
        limit_value=termination_latitude,
        use_as_lower_limit=False,
        terminate_exactly_on_final_condition=False
    )
    lower_lat_termination_settings = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=propagation_setup.dependent_variable.latitude("SUPER_SAT_37k", "Earth"),
        limit_value=-termination_latitude,
        use_as_lower_limit=True,
        terminate_exactly_on_final_condition=False
    )
    upper_long_termination_settings = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=propagation_setup.dependent_variable.longitude("SUPER_SAT_37k", "Earth"),
        limit_value=termination_longitude,
        use_as_lower_limit=False,
        terminate_exactly_on_final_condition=False
    )
    lower_long_termination_settings = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=propagation_setup.dependent_variable.longitude("SUPER_SAT_37k", "Earth"),
        limit_value=-termination_longitude,
        use_as_lower_limit=True,
        terminate_exactly_on_final_condition=False
    )

    hybrid_termination_conditions_list = [time_termination_settings, upper_lat_termination_settings, lower_lat_termination_settings, upper_long_termination_settings, lower_long_termination_settings]

    # Create termination settings object (when either the time of altitude condition is reached: propaation terminates)
    hybrid_termination_settings = propagation_setup.propagator.hybrid_termination(
        hybrid_termination_conditions_list,
        fulfill_single_condition=True
        )   
    
    ######################################################################################
    
    # ACCELERATIONS
    
    ######################################################################################
    
    acceleration_settings_on_vehicle = {
        # "Earth": [propagation_setup.acceleration.point_mass_gravity()],
        "Earth": [propagation_setup.acceleration.spherical_harmonic_gravity(10, 10)],
        "Moon": [propagation_setup.acceleration.point_mass_gravity()],
        "Sun": [propagation_setup.acceleration.point_mass_gravity()],
    }
    
    acceleration_settings = {'SUPER_SAT_37k': acceleration_settings_on_vehicle}
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies,
        acceleration_settings,
        ["SUPER_SAT_37k"],
        ["Earth"])
        
    ######################################################################################
    
    # PROPAGATOR CONDITIONS
    
    ######################################################################################
        
    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
            integrator_stepsize,
            integrator_coeff_set)
    
    mu = bodies.get("Earth").gravitational_parameter
    # T = 24*60**2
    T = constants.JULIAN_DAY

    # default_init_mee[0] = ((T/(2*np.pi))**2 * mu)**(1/3)
    
    cartesian_init_state = element_conversion.mee_to_cartesian(
        default_init_mee,
        mu,
        False
    )
    
    if decision_variable_dic:        
        delta_v = decision_variable_dic["dv_mag"] * decision_variable_dic["dv_unit_vect"]        
        cartesian_init_state[3:6] += delta_v
    
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies=["Earth"],
        acceleration_models=acceleration_models,
        bodies_to_integrate=["SUPER_SAT_37k"],
        initial_states=cartesian_init_state,
        initial_time=simulation_start_epoch,
        integrator_settings=integrator_settings,
        termination_settings=hybrid_termination_settings,
        propagator=propagation_setup.propagator.cowell,
        output_variables=default_dep_vars)
    
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies, propagator_settings)
    
    
    hf.save_dynamics_simulator_to_files(path_to_save_data, dynamics_simulator)
    hf.save_dependent_variable_info(dynamics_simulator.propagation_results.dependent_variable_ids, path_to_save_data + "/dependent_variable_ids.dat")
    
    
    
    pass

if __name__ == "__main__":
    
    decision_variable_dic = {}
    
    decision_variable_dic["dv_mag"] = -.1        
    decision_variable_dic["dv_unit_vect"] = np.array([0, 1, 0])    
    
    run_simulation("test2", 2 * 31 * 24 * 60**2, integrator_stepsize=600, decision_variable_dic=decision_variable_dic)


