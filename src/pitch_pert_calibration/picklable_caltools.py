import numpy as np
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pitch_pert_calibration.pitchpert_calibration import get_current_params, get_params_for_implementation
from pitch_pert_calibration.pitchpert_calibration import BlankParamsObject
from pitch_pert_calibration.pitchpert_calibration import truncate_data
from pitch_pert_calibration.pitchpert_calibration import DIVAController, Controller
from pitch_pert_calibration.pitchpert_calibration import AbsoluteSensorProcessor, RelativeSensorProcessor

import pickle
import copy
import os
import sys

# Global variables to hold the data needed by the objective function
_global_calibrator_data = None

def _initialize_global_data(params_obj, target_response, pert_signal, T_sim, 
                           truncate_start, truncate_end, sensor_processor):
    """Initialize global data for multiprocessing"""
    global _global_calibrator_data
    
    # Create a minimal data structure with only the essential data
    _global_calibrator_data = {
        'param_config': None,
        'system_type': params_obj.system_type,
        'kearney_name': getattr(params_obj, 'kearney_name', None),
        'arb_name': getattr(params_obj, 'arb_name', None),
        'ref_type': params_obj.ref_type,
        'dt': params_obj.dt,
        'target_response': target_response,
        'pert_signal': pert_signal,
        'T_sim': T_sim,
        'truncate_start': truncate_start,
        'truncate_end': truncate_end,
        'sensor_processor_type': type(sensor_processor).__name__,
        'cal_set_dict': getattr(params_obj, 'cal_set_dict', {}),
        # Store all params_obj attributes for 'expt config' feature
        'params_obj_attrs': {}
    }
    
    # Extract all attributes from params_obj (matching get_configs.py structure)
    # These are the same attributes that get_params() creates
    if params_obj.system_type == 'Template':
        template_attrs = [
            'actuator_delay', 'sensor_delay_aud', 'sensor_delay_som',
            'A', 'B', 'C_aud', 'C_som', 'K_aud', 'L_aud', 'Kf_aud', 
            'K_som', 'L_som', 'Kf_som'
        ]
        for attr in template_attrs:
            if hasattr(params_obj, attr):
                value = getattr(params_obj, attr)
                # Only store picklable values
                if isinstance(value, (int, float, str, list, dict, np.ndarray)):
                    _global_calibrator_data['params_obj_attrs'][attr] = value
                    
    elif params_obj.system_type == 'DIVA':
        diva_attrs = [
            'actuator_delay', 'sensor_delay_aud', 'sensor_delay_som',
            'tau_A', 'tau_S', 'tau_As', 'tau_Ss',
            'alpha_A', 'alpha_S', 'alpha_As', 'alpha_Ss', 'alpha_Av', 'alpha_Sv'
        ]
        for attr in diva_attrs:
            if hasattr(params_obj, attr):
                value = getattr(params_obj, attr)
                # Only store picklable values
                if isinstance(value, (int, float, str, list, dict, np.ndarray)):
                    _global_calibrator_data['params_obj_attrs'][attr] = value
    
    # Get parameter configuration
    if params_obj.system_type == 'DIVA':
        _global_calibrator_data['param_config'] = get_params_for_implementation(
            params_obj.system_type, kearney_name=params_obj.kearney_name
        )
    elif params_obj.system_type == 'Template':
        _global_calibrator_data['param_config'] = get_params_for_implementation(
            params_obj.system_type, arb_name=params_obj.arb_name
        )

def _standalone_objective_function(params, null_values_spec=None, layer=None):
    """
    Standalone objective function that can be pickled for multiprocessing.
    
    Args:
        params: 2D array of parameters where each row represents a particle's parameters
    
    Returns:
        Array of MSE values (one per particle) for PySwarms
    """
    global _global_calibrator_data
    
    if _global_calibrator_data is None:
        raise RuntimeError("Global data not initialized. Call _initialize_global_data first.")
    
    # Handle PySwarms vectorized evaluation (multiple particles)
    if isinstance(params, np.ndarray) and params.ndim == 2:
        costs = np.zeros(params.shape[0])
        
        for i in range(params.shape[0]):
            if layer == 'upper':
                particle_params = params[i]
                costs[i] = _evaluate_single_particle_standalone(particle_params, null_values_spec=null_values_spec)
            else:
                particle_params = params[i]
                costs[i] = _evaluate_single_particle_standalone(particle_params, null_values_spec=null_values_spec)
            
            # Handle infinite values
            if np.isinf(costs[i]):
                costs[i] = 1e10  # Use a large finite value instead of inf
        
        return costs
    else:
        # Handle single particle evaluation
        cost = _evaluate_single_particle_standalone(params, null_values_spec=null_values_spec)
        if np.isinf(cost):
            cost = 1e10
        return cost

def _evaluate_single_particle_standalone(params, null_values_spec=None):
    """
    Evaluate a single particle's parameters (standalone version).
    
    Args:
        params: 1D array of parameters for a single particle
    
    Returns:
        MSE value for this particle
    """
    global _global_calibrator_data

    if null_values_spec is not None:
        null_values = null_values_spec
    else:
        null_values = _global_calibrator_data['cal_set_dict']['null_values']
    
    # Create current_params dict from the parameter array
    temp_param_dict = {}
    
    for i, param_name in enumerate(_global_calibrator_data['param_config']):
        if i < len(params):
            temp_param_dict[param_name] = params[i]
    
    # Create a minimal params object for simulation
    temp_params = BlankParamsObject(**temp_param_dict)
    
    # Recreate the original params_obj structure for get_current_params
    # This matches the structure created by get_params() in get_configs.py
    perm_param_dict = {}
    perm_param_dict['system_type'] = _global_calibrator_data['system_type']
    perm_param_dict['kearney_name'] = _global_calibrator_data['kearney_name']
    perm_param_dict['arb_name'] = _global_calibrator_data['arb_name']
    perm_param_dict['cal_set_dict'] = _global_calibrator_data['cal_set_dict']
    perm_param_dict['ref_type'] = _global_calibrator_data['ref_type']
    perm_param_dict['dt'] = _global_calibrator_data['dt']
    perm_param_dict['target_response'] = _global_calibrator_data['target_response']

    for key, value in _global_calibrator_data['params_obj_attrs'].items():
        perm_param_dict[key] = value
    
    recreated_params_obj = BlankParamsObject(**perm_param_dict)
    
    # Get current parameters using the calibration function
    current_params = get_current_params(
        recreated_params_obj,  # Use recreated params_obj instead of simple_obj
        _global_calibrator_data['param_config'], 
        cal_only=True, 
        null_values=null_values, 
        params=temp_params
    )
    
    # Create sensor processor
    if _global_calibrator_data['sensor_processor_type'] == 'AbsoluteSensorProcessor':
        sensor_processor = AbsoluteSensorProcessor()
    else:
        sensor_processor = RelativeSensorProcessor()
    
    # Run simulation based on system type
    if _global_calibrator_data['system_type'] == 'DIVA':
        system = DIVAController(
            sensor_processor, 
            _global_calibrator_data['T_sim'], 
            _global_calibrator_data['dt'], 
            _global_calibrator_data['pert_signal'].signal, 
            _global_calibrator_data['pert_signal'].start_ramp_up, 
            _global_calibrator_data['target_response'], 
            current_params
        )
        system.simulate(_global_calibrator_data['kearney_name'])
    else:
        # Template system
        system = Controller(
            sensor_processor=sensor_processor, 
            params=current_params,
            ref_type=_global_calibrator_data['ref_type'], 
            dist_custom=_global_calibrator_data['pert_signal'].signal,
            dist_type=['Auditory'],
            timeseries=_global_calibrator_data['T_sim']
        )
        
        # Run simulation
        system.simulate_with_2sensors(
            delta_t_s_aud=int(current_params['sensor_delay_aud']),
            delta_t_s_som=int(current_params['sensor_delay_som']),
            delta_t_a=int(current_params['actuator_delay'])
        )
    
    # Calculate MSE between simulation and calibration data
    timeseries_truncated, system_response_truncated = truncate_data(
        _global_calibrator_data['T_sim'], 
        system.x, 
        _global_calibrator_data['truncate_start'], 
        _global_calibrator_data['truncate_end']
    )
    
    mse = system.mse(system_response_truncated, _global_calibrator_data['target_response'], check_stability=True, full_data2check=system.x)
    return mse

    # _evaluate_upper_particle_standalone(params, null_values_spec=null_values_spec)
    # #run a lower layer optimization for the input particle


    