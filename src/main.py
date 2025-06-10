import numpy as np
from controllers.base import ControlSystem
from controllers.implementations import AbsEstController, RelEstController
from utils.analysis import AnalysisMixin
from visualization.plotting import PlotMixin


# Simulation parameters
A = np.array([0.5])
B = np.array([1])
C_aud = np.array([1])
C_som = np.array([1])

# Run simulation
ref_types = ['sin'] #'sin' or 'null'
archs = ['q_based'] #['x_based','q_based']
actuator_delays = 1
sensor_delays = 3

for ref_item in ref_types:
    print('Running simulation with reference type:', ref_item)
    for arch_item in archs:
        print('Running simulation with architecture:', arch_item)
        if arch_item == 'x_based':
            system = AbsEstController(A, B, C_aud, ref_type=ref_item, K_vals=[0.5, 0.2], L_vals=[0.5, 0.5])    
        elif arch_item == 'q_based':
            system = RelEstController(A, B, C_aud, ref_type=ref_item, dist_type=['Auditory'], K_vals=[0.5, 0.2], L_vals=[0.5, 0.5])
        else:
            raise ValueError('Invalid architecture:', arch_item)

        #system.simulate_with_1sensor(delta_t_s=sensor_delays, delta_t_a=actuator_delays)
        system.simulate_with_2sensors(delta_t_s_aud=sensor_delays, delta_t_s_som=sensor_delays, delta_t_a=actuator_delays)
        print(arch_item, ref_item, 'RMSE:', system.rmse())
        system.plot_transient(arch_item) 
        system.plot_all(arch_item)