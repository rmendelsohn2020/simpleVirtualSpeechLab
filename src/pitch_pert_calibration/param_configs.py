# Centralized parameter configurations with implementation-specific parameter sets
PARAM_CONFIGS = {
    'DIVA': {
        'title': 'Optimized DIVA ({kearney_name}) parameters:',
        'param_sets': {
            # D1: Only uses alpha_A (alpha_S = 0)
            'D1': [
                'alpha_A', 
                'alpha_S', 
                'tau_A'
            ],
            # D2: Uses both alpha_A and alpha_S
            'D2': [
                'alpha_A',
                'alpha_S',
                'tau_A',
                'tau_S'
            ],
            # D5-D10: Uses different parameter combinations (placeholder for now)
            'D3': [
                'alpha_A',
                'alpha_Av',
                'tau_A'
            ],
            'D4': [
                'alpha_A',
                'alpha_Av', 
                'tau_A',
                'tau_Av'
            ],
            'D5': [
                'alpha_A',
                'alpha_S',
                'alpha_Av',
                'tau_A'
            ],
            'D6': [
                'alpha_A',
                'alpha_S',
                'alpha_Av',
                'tau_A',
                'tau_S'
            ],
            'D7': [
                'alpha_A',
                'alpha_S',
                'alpha_Av',
                'tau_A',
                'tau_S',
                'tau_Av'
            ],
            'D8': [
                'alpha_A',
                'alpha_S',
                'alpha_Av',
                'alpha_Sv',
                'tau_A',
                'tau_Av'
            ],
            # D11-D15: Uses different parameter combinations (placeholder for now)
            'D9': [
                'alpha_A',
                'alpha_S',
                'alpha_Av',
                'alpha_Sv',
                'tau_A',
                'tau_S',
                'tau_Av',
                'tau_Sv'
            ],
            'D10': [
                'alpha_A',
                'alpha_S',
                'alpha_Av',
                'alpha_Sv',
                'tau_A',
                'tau_S',
                'tau_Av',
                'tau_Sv'
            ],
            'D11': [
                'alpha_A',
                'alpha_As',
                'tau_A',
                'tau_As'
            ],
            'D12': [
                'alpha_A',
                'alpha_S',
                'alpha_As',
                'tau_A',
                'tau_S',
                'tau_As'
            ],
            'D13': [
                'alpha_A',
                'alpha_S',
                'alpha_As',
                'alpha_Ss',
                'tau_A',
                'tau_As',
                'tau_Ss'
            ],
            'D14': [
                'alpha_A',
                'alpha_S',
                'alpha_As',
                'alpha_Ss',
                'tau_A',
                'tau_S',
                'tau_As'
            ],
            'D15': [
                'alpha_A',
                'alpha_S',
                'alpha_As',
                'alpha_Ss',
                'tau_A',
                'tau_S',
                'tau_As',
                'tau_Ss'
            ]
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
        'param_sets': {
            'all': [
            'A',
            'B',
            'C_aud',
            'C_som',
            'K_aud',
            'L_aud',
            'K_som',
            'L_som',
            'sensor_delay_aud',
            'sensor_delay_som',
            'actuator_delay'
            ],
            'upper layer': [
                'A',
                'B',
                'C_aud',
                'C_som'
            ],
            'lower layer': [
                'K_aud',
                'K_som',
                'L_aud',
                'L_som',
                'sensor_delay_aud',
                'sensor_delay_som',
                'actuator_delay'
            ],
            'T1': [
                'K_aud',
                'K_som',
                'sensor_delay_aud'
            ],
            'T2': [
                'K_aud',
                'K_som',
                'sensor_delay_aud',
                'sensor_delay_som'
            ],
            'T3': [
                'K_aud',
                'K_som',
                'sensor_delay_aud',
                'sensor_delay_som',
                'actuator_delay'
            ],
            'T4': [
                'A',
                'B',
                'K_aud',
                'K_som',
                'sensor_delay_aud',
                'sensor_delay_som',
                'actuator_delay'
            ],
            'T5': [
                'A',
                'B',
                'C_aud',
                'K_aud',
                'K_som',
                'sensor_delay_aud',
                'sensor_delay_som',
                'actuator_delay'
            ],
            'T6': [
                'A',
                'B',
                'C_aud',
                'K_aud',
                'K_som',
                'sensor_delay_aud'
            ],
            'T7': [
                'K_aud',
                'K_som',
                'L_aud',
                'L_som'
            ],
            'T8': [
                'A',
                'B',
                'C_aud',
                'C_som',
                'K_aud',
                'K_som',
                'L_aud',
                'L_som'
            ],
            'T9': [
                'A',
                'B',
                'C_aud',
                'C_som',
                'K_aud',
            ]
            
        }
    }
}

# Add this after PARAM_CONFIGS
PARAM_BOUNDS = {
    'DIVA': {
        'alpha_A': (1e-6, 5),
        'alpha_S': (1e-6, 5),
        'alpha_Av': (1e-6, 5),
        'alpha_Sv': (1e-6, 5),
        'alpha_As': (1e-6, 5),
        'alpha_Ss': (1e-6, 5),
        'tau_A': (0, 20),
        'tau_S': (0, 20),
        'tau_Av': (0, 20),
        'tau_Sv': (0, 20),
        'tau_As': (0, 20),
        'tau_Ss': (0, 20)
    },
    'Template': {
        'A': (1e-6, 3),
        'B': (1e-6, 5),
        'C_aud': (1e-6, 5),
        'C_som': (1e-6, 5),
        'K_aud': (1e-6, 6),
        'L_aud': (1e-6, 6),
        'K_som': (1e-6, 6),
        'L_som': (1e-6, 6),
        'sensor_delay_aud': (0, 20),
        'sensor_delay_som': (0, 20),
        'actuator_delay': (0, 20)
    }    
}
PARAM_NULL_VALUES = {
    'DIVA': {
        'alpha_A': 1,
        'alpha_S': 1,
        'alpha_Av': 1,
        'alpha_Sv': 1,
        'alpha_As': 1,
        'alpha_Ss': 1,
        'tau_A': 0,
        'tau_S': 0,
        'tau_Av': 0,
        'tau_Sv': 0,
        'tau_As': 0,
        'tau_Ss': 0
    },
    'Template': {
        'A': 1,
        'B': 1,
        'C_aud': 1,
        'C_som': 1,
        'K_aud': 1,
        'L_aud': 1,
        'K_som': 1,
        'L_som': 1,
        'sensor_delay_aud': 0,
        'sensor_delay_som': 0,
        'actuator_delay': 0
    }
}
