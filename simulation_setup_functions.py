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


def get_general_termination_settings(termination_latitude, termination_longitude):
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

    return [upper_lat_termination_settings, lower_lat_termination_settings, upper_long_termination_settings,
            lower_long_termination_settings]


def get_integrator_settings(integrator_stepsize, integrator_coeff_set):
    return propagation_setup.integrator.runge_kutta_fixed_step_size(
            integrator_stepsize,
            integrator_coeff_set)


