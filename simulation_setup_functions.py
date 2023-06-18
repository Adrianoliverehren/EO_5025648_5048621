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


def get_general_termination_settings(termination_latitude, termination_longitude, max_cpu_time):
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

    cpu_termination_setting = propagation_setup.propagator.cpu_time_termination(max_cpu_time)

    return [upper_lat_termination_settings, lower_lat_termination_settings, upper_long_termination_settings,
            lower_long_termination_settings, cpu_termination_setting]


def get_integrator_settings(integrator_settings_dic):
    
    if integrator_settings_dic["type"] == "multistage.fixed":
        integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
            integrator_settings_dic["step_size"],
            integrator_settings_dic["integrator_coeff_set"])
    
    
        
    return integrator_settings


def get_acceleration_settings(bodies):
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

    return acceleration_models

if __name__ == "__main__":
    default_integrator_settings_dic = {
        "type": "multistage.fixed",
        "step_size": 1200,
        "integrator_coeff_set": propagation_setup.integrator.CoefficientSets.rkf_45
    }
    
    print(get_integrator_settings(default_integrator_settings_dic))