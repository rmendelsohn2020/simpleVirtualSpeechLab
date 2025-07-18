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
        'params': [
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
        ]
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
        'tau_A': (1e-6, 5),
        'tau_S': (1e-6, 5),
        'tau_Av': (1e-6, 5),
        'tau_Sv': (1e-6, 5),
        'tau_As': (1e-6, 5),
        'tau_Ss': (1e-6, 5)
    },
    'Template': {
        'A': (0, 3),
        'B': (0, 5),
        'C_aud': (0, 5),
        'C_som': (0, 5),
        'K_aud': (0, 10),
        'L_aud': (0, 10),
        'K_som': (0, 10),
        'L_som': (0, 10),
        'sensor_delay_aud': (0, 20),
        'sensor_delay_som': (0, 20),
        'actuator_delay': (0, 20)
    }
}
