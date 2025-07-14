from utils.get_configs import get_paths
from datetime import datetime
import numpy as np

# Centralized parameter configurations with implementation-specific parameter sets
PARAM_CONFIGS = {
    'DIVA': {
        'title': 'Optimized DIVA ({kearney_name}) parameters:',
        'param_sets': {
            # D1: Only uses alpha_A (alpha_S = 0)
            'D1': [
                ('alpha_A', 'alpha_A_init'),
                ('alpha_Av', 'alpha_Av_init'),
                ('alpha_Sv', 'alpha_Sv_init'),
                ('tau_A', 'tau_A_init'),
                ('tau_S', 'tau_S_init'),
                ('tau_As', 'tau_As_init'),
                ('tau_Ss', 'tau_Ss_init'),
            ],
            # D2: Uses both alpha_A and alpha_S
            'D2': [
                ('alpha_A', 'alpha_A_init'),
                ('alpha_S', 'alpha_S_init'),
                ('alpha_Av', 'alpha_Av_init'),
                ('alpha_Sv', 'alpha_Sv_init'),
                ('tau_A', 'tau_A_init'),
                ('tau_S', 'tau_S_init'),
                ('tau_As', 'tau_As_init'),
                ('tau_Ss', 'tau_Ss_init'),
            ],
            # D5-D10: Uses different parameter combinations (placeholder for now)
            'D5': [
                ('alpha_A', 'alpha_A_init'),
                ('alpha_S', 'alpha_S_init'),
                ('alpha_Av', 'alpha_Av_init'),
                ('alpha_Sv', 'alpha_Sv_init'),
                ('tau_A', 'tau_A_init'),
                ('tau_S', 'tau_S_init'),
                ('tau_As', 'tau_As_init'),
                ('tau_Ss', 'tau_Ss_init'),
            ],
            'D6': [
                ('alpha_A', 'alpha_A_init'),
                ('alpha_S', 'alpha_S_init'),
                ('alpha_Av', 'alpha_Av_init'),
                ('alpha_Sv', 'alpha_Sv_init'),
                ('tau_A', 'tau_A_init'),
                ('tau_S', 'tau_S_init'),
                ('tau_As', 'tau_As_init'),
                ('tau_Ss', 'tau_Ss_init'),
            ],
            'D7': [
                ('alpha_A', 'alpha_A_init'),
                ('alpha_S', 'alpha_S_init'),
                ('alpha_Av', 'alpha_Av_init'),
                ('alpha_Sv', 'alpha_Sv_init'),
                ('tau_A', 'tau_A_init'),
                ('tau_S', 'tau_S_init'),
                ('tau_As', 'tau_As_init'),
                ('tau_Ss', 'tau_Ss_init'),
            ],
            'D8': [
                ('alpha_A', 'alpha_A_init'),
                ('alpha_S', 'alpha_S_init'),
                ('alpha_Av', 'alpha_Av_init'),
                ('alpha_Sv', 'alpha_Sv_init'),
                ('tau_A', 'tau_A_init'),
                ('tau_S', 'tau_S_init'),
                ('tau_As', 'tau_As_init'),
                ('tau_Ss', 'tau_Ss_init'),
            ],
            'D9': [
                ('alpha_A', 'alpha_A_init'),
                ('alpha_S', 'alpha_S_init'),
                ('alpha_Av', 'alpha_Av_init'),
                ('alpha_Sv', 'alpha_Sv_init'),
                ('tau_A', 'tau_A_init'),
                ('tau_S', 'tau_S_init'),
                ('tau_As', 'tau_As_init'),
                ('tau_Ss', 'tau_Ss_init'),
            ],
            'D10': [
                ('alpha_A', 'alpha_A_init'),
                ('alpha_S', 'alpha_S_init'),
                ('alpha_Av', 'alpha_Av_init'),
                ('alpha_Sv', 'alpha_Sv_init'),
                ('tau_A', 'tau_A_init'),
                ('tau_S', 'tau_S_init'),
                ('tau_As', 'tau_As_init'),
                ('tau_Ss', 'tau_Ss_init'),
            ],
            # D11-D15: Uses different parameter combinations (placeholder for now)
            'D11': [
                ('alpha_A', 'alpha_A_init'),
                ('alpha_S', 'alpha_S_init'),
                ('alpha_Av', 'alpha_Av_init'),
                ('alpha_Sv', 'alpha_Sv_init'),
                ('tau_A', 'tau_A_init'),
                ('tau_S', 'tau_S_init'),
                ('tau_As', 'tau_As_init'),
                ('tau_Ss', 'tau_Ss_init'),
            ],
            'D12': [
                ('alpha_A', 'alpha_A_init'),
                ('alpha_S', 'alpha_S_init'),
                ('alpha_Av', 'alpha_Av_init'),
                ('alpha_Sv', 'alpha_Sv_init'),
                ('tau_A', 'tau_A_init'),
                ('tau_S', 'tau_S_init'),
                ('tau_As', 'tau_As_init'),
                ('tau_Ss', 'tau_Ss_init'),
            ],
            'D13': [
                ('alpha_A', 'alpha_A_init'),
                ('alpha_S', 'alpha_S_init'),
                ('alpha_Av', 'alpha_Av_init'),
                ('alpha_Sv', 'alpha_Sv_init'),
                ('tau_A', 'tau_A_init'),
                ('tau_S', 'tau_S_init'),
                ('tau_As', 'tau_As_init'),
                ('tau_Ss', 'tau_Ss_init'),
            ],
            'D14': [
                ('alpha_A', 'alpha_A_init'),
                ('alpha_S', 'alpha_S_init'),
                ('alpha_Av', 'alpha_Av_init'),
                ('alpha_Sv', 'alpha_Sv_init'),
                ('tau_A', 'tau_A_init'),
                ('tau_S', 'tau_S_init'),
                ('tau_As', 'tau_As_init'),
                ('tau_Ss', 'tau_Ss_init'),
            ],
            'D15': [
                ('alpha_A', 'alpha_A_init'),
                ('alpha_S', 'alpha_S_init'),
                ('alpha_Av', 'alpha_Av_init'),
                ('alpha_Sv', 'alpha_Sv_init'),
                ('tau_A', 'tau_A_init'),
                ('tau_S', 'tau_S_init'),
                ('tau_As', 'tau_As_init'),
                ('tau_Ss', 'tau_Ss_init'),
            ],
        },
        'bounds': [
            (1e-6, 5),
            (1e-6, 5),
            (1e-6, 5),
            (1e-6, 5),
            (1e-6, 5),
            (1e-6, 5),
            (1e-6, 5),
            (1e-6, 5)
        ]
    },
    'Template': {
        'title': 'Optimized Template parameters:',
        'params': [
            ('A', 'A_init'),
            ('B', 'B_init'),
            ('C_aud', 'C_aud_init'),
            ('C_som', 'C_som_init'),
            ('K_aud', 'K_aud_init'),
            ('L_aud', 'L_aud_init'),
            ('K_som', 'K_som_init'),
            ('L_som', 'L_som_init'),
        ]
    }
}

def get_params_for_implementation(system_type, kearney_name=None):
    """
    Get the appropriate parameter list for a given system type and implementation.
    
    Args:
        system_type: 'DIVA' or 'Template'
        kearney_name: Implementation name (e.g., 'D1', 'D2', etc.) for DIVA
    
    Returns:
        List of (param_name, attr_name) tuples for the specific implementation
    """
    config = PARAM_CONFIGS.get(system_type)
    if not config:
        return []
    
    if system_type == 'DIVA':
        if kearney_name and 'param_sets' in config:
            return config['param_sets'].get(kearney_name, [])
        else:
            # Fallback to default if no specific implementation found
            return config.get('params', [])
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
    else:
        params_list = get_params_for_implementation(cal_params.system_type)

    # Get title with kearney_name if available
    title = config['title']
    if hasattr(cal_params, 'kearney_name'):
        title = title.format(kearney_name=cal_params.kearney_name)
    else:
        title = title.format(kearney_name='')

    if 'print' in format_opt:
        print(title)
        for param_name, attr_name in params_list:
            if hasattr(cal_params, attr_name):
                print(f'{param_name}: {getattr(cal_params, attr_name)}')

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
            
            for param_name, attr_name in params_list:
                if hasattr(cal_params, attr_name):
                    value = getattr(cal_params, attr_name)
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