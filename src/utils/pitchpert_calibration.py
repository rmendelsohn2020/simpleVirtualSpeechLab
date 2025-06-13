import pandas as pd
import numpy as np
from scipy.optimize import minimize
from controllers.implementations import Controller, AbsoluteSensorProcessor, RelativeSensorProcessor
from utils.analysis import AnalysisMixin
from visualization.plotting import PlotMixin
from utils.pitchpert_dataprep import truncate_data

def get_perturbation_event_times(file_path, epsilon=1e-10):
    df = pd.read_csv(file_path)

    time_col=0
    perturb_col=1

    # Get the series
    time = df.iloc[:, time_col]
    perturbation = df.iloc[:, perturb_col]

    # Find index of first non-zero perturbation using epsilon tolerance
    first_nonzero_idx = (perturbation[abs(perturbation) > epsilon].index[0]) - 1
    print('first_nonzero_idx', first_nonzero_idx)

    # Find index of maximum absolute value (peak of perturbation)
    max_abs_val = abs(perturbation).max()
    print('max_abs_val', max_abs_val)
    # Find the actual value (could be negative) at the maximum absolute point
    max_idx = (perturbation[abs(perturbation) > (max_abs_val - epsilon)].index[0])
    print('max_idx', max_idx)
    # Get times
    time_at_first_nonzero = time.iloc[first_nonzero_idx]
    time_at_maximum = time.iloc[max_idx]

    return time_at_first_nonzero, time_at_maximum

class PitchPertCalibrator:
    def __init__(self, params_obj, target_response, pert_signal, T_sim, truncate_start, truncate_end):
        """
        Initialize the calibrator with system parameters and data.
        
        Args:
            params_obj: Object containing system parameters
            target_response: Target response data to match
            pert_signal: Perturbation signal
            T_sim: Time series for simulation
            truncate_start: Start time for truncation
            truncate_end: End time for truncation
        """
        self.params_obj = params_obj
        self.target_response = target_response
        self.pert_signal = pert_signal
        self.T_sim = T_sim
        self.truncate_start = truncate_start
        self.truncate_end = truncate_end
        self.mse_history = []

    def objective_function(self, params):
        """
        Objective function for parameter optimization.
        
        Args:
            params: Flattened array of parameters to optimize
        
        Returns:
            MSE between simulation and target response
        """
        # Unpack parameters
        A = np.array([params[0]])
        B = np.array([params[1]])
        C_aud = np.array([params[2]])
        C_som = np.array([params[3]])
        K_aud = params[4]
        L_aud = params[5]
        K_som = params[6]
        L_som = params[7]
        sensor_delay_aud = params[8]
        sensor_delay_som = params[9]
        actuator_delay = params[10]
        
        # Create system with current parameters
        system = Controller(
            sensor_processor=AbsoluteSensorProcessor(), 
            input_A=A, input_B=B, input_C=C_aud, 
            ref_type=self.params_obj.ref_type, 
            K_vals=[K_aud, K_som], 
            L_vals=[L_aud, L_som],
            dist_custom=self.pert_signal.signal,
            dist_type=['Auditory'],
            timeseries=self.T_sim
        )
        
        # Run simulation
        system.simulate_with_2sensors(
            delta_t_s_aud=int(sensor_delay_aud),
            delta_t_s_som=int(sensor_delay_som),
            delta_t_a=int(actuator_delay)
        )
        
        # Calculate MSE between simulation and calibration data
        timeseries_truncated, system_response_truncated = truncate_data(
            self.T_sim, system.x, self.truncate_start, self.truncate_end
        )
        return system.mse(system_response_truncated, self.target_response)

    def callback_function(self, xk, *args):
        """
        Callback function to track optimization progress.
        
        Args:
            xk: Current parameter values
            *args: Additional arguments that may be passed by different optimization methods
        """
        current_mse = self.objective_function(xk)
        self.mse_history.append(current_mse)
        return False

    def calibrate(self, max_iterations=200, learning_rate=0.01, tolerance=1e-6):
        """
        Calibrate system parameters to match calibration data using scipy.optimize.minimize.
        
        Args:
            max_iterations: Maximum number of optimization iterations
            learning_rate: Step size for parameter updates (not used in scipy minimize)
            tolerance: Convergence threshold for mse improvement
        
        Returns:
            Optimized parameters object and optimization history
        """
        # Define parameter bounds
        bounds = [
            (1e-6, 1.1),      # A
            (1e-6, 5.0),      # B
            (1e-6, 5.0),      # C_aud
            (1e-6, 5.0),      # C_som
            (1e-6, 1.0),      # K_aud
            (1e-6, 1.0),      # L_aud
            (1e-6, 1.0),      # K_som
            (1e-6, 1.0),      # L_som
            (0.0, 10.0),      # sensor_delay_aud
            (0.0, 10.0),      # sensor_delay_som
            (0.0, 10.0)       # actuator_delay
        ]
        
        # Initial parameter values
        x0 = [
            self.params_obj.A_init[0],
            self.params_obj.B_init[0],
            self.params_obj.C_aud_init[0],
            self.params_obj.C_som_init[0],
            self.params_obj.K_aud_init,
            self.params_obj.L_aud_init,
            self.params_obj.K_som_init,
            self.params_obj.L_som_init,
            self.params_obj.sensor_delay_aud,
            self.params_obj.sensor_delay_som,
            self.params_obj.actuator_delay
        ]
        
        # Reset MSE history
        self.mse_history = []

        # Run optimization
        result = minimize(
            self.objective_function,
            x0,
            method='trust-constr',
            bounds=bounds,
            options={'maxiter': max_iterations},
            callback=self.callback_function
        )
        
        # Update parameters object with optimized values
        self.params_obj.A_init = np.array([result.x[0]])
        self.params_obj.B_init = np.array([result.x[1]])
        self.params_obj.C_aud_init = np.array([result.x[2]])
        self.params_obj.C_som_init = np.array([result.x[3]])
        self.params_obj.K_aud_init = result.x[4]
        self.params_obj.L_aud_init = result.x[5]
        self.params_obj.K_som_init = result.x[6]
        self.params_obj.L_som_init = result.x[7]
        self.params_obj.sensor_delay_aud = result.x[8]
        self.params_obj.sensor_delay_som = result.x[9]
        self.params_obj.actuator_delay = result.x[10]
        
        print('Optimization completed:')
        print(f'Final MSE: {result.fun}')
        print(f'Number of iterations: {result.nit}')
        print(f'Optimization success: {result.success}')
        print(f'Message: {result.message}')
        print(f'Number of MSE values recorded: {len(self.mse_history)}')
        
        return self.params_obj, self.mse_history