from utils.get_configs import get_paths
from datetime import datetime
import numpy as np
from pitch_pert_calibration.param_configs import PARAM_CONFIGS, PARAM_BOUNDS, PARAM_NULL_VALUES

class BlankParamsObject:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def calibration_info_pack(params_obj, cal_only=False, print_opt=['print'], custom_label='', null_values=None):
    if params_obj.system_type == 'DIVA':
        param_config = get_params_for_implementation(params_obj.system_type, params_obj.kearney_name, null_values=null_values)
        print('param_config', param_config)
    elif params_obj.system_type == 'Template':
        param_config = get_params_for_implementation(params_obj.system_type, arb_name=params_obj.arb_name, null_values=null_values)
        print('param_config', param_config)
    else:
        param_config = get_params_for_implementation(params_obj.system_type)
        print('WARNING: Unlisted system type')
    param_bounds = get_bounds_for_params(params_obj.system_type, param_config)
    
    current_params = get_current_params(params_obj, param_config, cal_only=cal_only, null_values=null_values)
    x0 = get_init_values_for_params(current_params, param_config)

    
    if 'print' in print_opt:
        print(f'====INFO {custom_label}====')
        print('cal_only', cal_only)
        print('param_config', param_config)
        print('param_bounds', param_bounds)
        print('x0', x0)
        print(f'{custom_label} current_params', current_params)
        print(f'====END {custom_label} INFO====')
    
    return param_config, param_bounds, x0, current_params

def get_current_params(params_obj, param_config, cal_only=False, null_values=None, params=None):
    current_params = {}
    
    if cal_only:
        if params_obj.system_type == 'DIVA':
            params_for_cal = get_params_for_implementation(params_obj.system_type, params_obj.kearney_name)
        elif params_obj.system_type == 'Template':
            params_for_cal = get_params_for_implementation(params_obj.system_type, arb_name=params_obj.arb_name, null_values=null_values)
        else:
            params_for_cal = get_params_for_implementation(params_obj.system_type)
            print('WARNING: Unlisted system type')
        param_config = params_for_cal

    if params is None:
            params = params_obj
            print('Only one parameter source provided')
    print(f'get_current_params using: {params}')
    for param_name in param_config:
        if hasattr(params, param_name):
            current_params[param_name] = getattr(params, param_name)
            print(f'Listed {param_name}: {current_params[param_name]}')
        elif null_values=='null':
            current_params[param_name] = PARAM_NULL_VALUES[params_obj.system_type][param_name]
            print(f'Null {param_name}: {current_params[param_name]}')
        elif null_values=='expt config':
            current_params[param_name] = getattr(params_obj, param_name)
            print(f'Expt config {param_name}: {current_params[param_name]}')
        else:
            print('WARNING: Unlisted null_values option')

    return current_params

def get_init_values_for_params(params_obj, params_list):
    x0 = [getattr(params_obj, param_name, 1.0) for param_name in params_list]

def get_bounds_for_params(system_type, param_names):
    """Get bounds for specific parameters in order."""
    bounds_config = PARAM_BOUNDS.get(system_type, {})
    return [bounds_config.get(param_name, (0, 1)) for param_name in param_names]

def get_params_for_implementation(system_type, kearney_name=None, arb_name=None, null_values=None):
    """
    Get the appropriate parameter list for a given system type and implementation.
    
    Args:
        system_type: 'DIVA' or 'Template'
        kearney_name: Implementation name (e.g., 'D1', 'D2', etc.) for DIVA
        arb_name: Articulatory name (e.g., 'all', 'T1', 'T2', etc.) for Template
    Returns:
        List of (param_name, attr_name) tuples for the specific implementation
    """
    config = PARAM_CONFIGS.get(system_type)
    if not config:
        print(f"Unknown system type: {system_type}")
        return []
    
    if system_type == 'DIVA':
        return config['param_sets'].get(kearney_name, [])
    elif system_type == 'Template':
        if null_values is not None:
            print(f"Using null values for {arb_name}")
            
            config_params = config['param_sets'].get('all', [])
            print(f"config_params: {config_params}")
            return config_params
        else:
            print(f"Using {arb_name} values")
            config_params = config['param_sets'].get(arb_name, [])
            print(f"config_params: {config_params}")
            return config_params
        # if null_values:
        #     print('Using null values for params:')
        #     for param_name in param_config:
        #         if param_name not in current_params:
        #             current_params[param_name] = PARAM_NULL_VALUES[params_obj.system_type][param_name]
        #             print(f'{param_name}: {current_params[param_name]}')
            
    else:
        return config.get('params', [])

def readout_optimized_params(cal_params, sensor_delay_aud=None, sensor_delay_som=None, actuator_delay=None, format_opt=['txt','print'], output_dir=None):
    if output_dir is None:
        path_obj = get_paths()
        output_dir = path_obj.fig_save_path

    config = PARAM_CONFIGS.get(cal_params.system_type)
    if not config:
        print(f"Unknown system type: {cal_params.system_type}")
        return

    # Get the appropriate parameter list for this implementation
    if cal_params.system_type == 'DIVA' and hasattr(cal_params, 'kearney_name'):
        params_list = get_params_for_implementation(cal_params.system_type, cal_params.kearney_name)
    elif cal_params.system_type == 'Template' and hasattr(cal_params, 'arb_name'):
        params_list = get_params_for_implementation(cal_params.system_type, arb_name=cal_params.arb_name)
    else:
        params_list = get_params_for_implementation(cal_params.system_type)
        print('WARNING:Unlisted system type')
   
    # Get title with kearney_name if available
    title = config['title']
    if hasattr(cal_params, 'kearney_name'):
        title = title.format(kearney_name=cal_params.kearney_name)
    else:
        title = title.format(kearney_name='')

    

    if 'print' in format_opt:
        print(title)
        for param_name in params_list:
            if hasattr(cal_params, param_name):
                print(f'{param_name}: {getattr(cal_params, param_name)}')

        print(f'sensor_delay_aud: {sensor_delay_aud}')
        print(f'sensor_delay_som: {sensor_delay_som}')
        print(f'actuator_delay: {actuator_delay}')

   
        

    if 'txt' in format_opt:
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        param_save_path = f"{output_dir}/optimized_params_{timestamp}.txt"

        with open(param_save_path, 'w') as f:
            f.write(f"{title}\n")
            f.write("-----------------\n")
            
            for param_name in params_list:
                if hasattr(cal_params, param_name):
                    value = getattr(cal_params, param_name)
                    # Handle array parameters differently
                    if isinstance(value, (list, np.ndarray)):
                        f.write(f"{param_name}:\n{value}\n\n")
                    else:
                        f.write(f"{param_name}: {value}\n")
            
            # Add delay information
            if cal_params.system_type == 'Template':
                f.write(f"Auditory sensor delay: {sensor_delay_aud}\n")
                f.write(f"Somatosensory sensor delay: {sensor_delay_som}\n")
                f.write(f"Actuator delay: {actuator_delay}\n")

        print(f"Optimized parameters saved to {param_save_path}")