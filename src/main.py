import numpy as np
from controllers.base import ControlSystem
from controllers.implementations import AbsEstController, RelEstController
from utils.analysis import AnalysisMixin
from visualization.plotting import PlotMixin


# Simulation parameters
A = np.array([0.5])
B = np.array([1])
C = np.array([1])

# Run simulation
ref_types = ['sin'] #'sin' or 'null'
archs = ['q_based'] #['x_based','q_based']
actuator_delays = 0
sensor_delays = 3

for ref_item in ref_types:
    print('Running simulation with reference type:', ref_item)
    for arch_item in archs:
        print('Running simulation with architecture:', arch_item)
        if arch_item == 'x_based':
            system = AbsEstController(A, B, C, ref_item)    
        elif arch_item == 'q_based':
            system = RelEstController(A, B, C, ref_item)
        else:
            raise ValueError('Invalid architecture:', arch_item)

        system.simulate_with_2sensors()
        print(arch_item, ref_item, 'RMSE:', system.rmse())
        system.plot_transient(arch_item) 