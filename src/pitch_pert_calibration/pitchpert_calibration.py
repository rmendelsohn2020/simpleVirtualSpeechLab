import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
import json
from datetime import datetime
import random
import psutil
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from controllers.base import ControlSystem
from controllers.implementations import Controller, AbsoluteSensorProcessor, RelativeSensorProcessor
from utils.analysis import AnalysisMixin
from visualization.plotting import PlotMixin
from pitch_pert_calibration.pitchpert_dataprep import data_prep, truncate_data
from utils.signal_synth import RampedStep1D
from utils.get_configs import get_paths, get_params
from controllers.simpleDIVAtest import Controller as DIVAController
from controllers.simpleDIVAtest import get_sensor_processor
from visualization.readouts import get_params_for_implementation, readout_optimized_params, get_current_params, calibration_info_pack, BlankParamsObject
from utils.processing import make_jsonable_dict
from pitch_pert_calibration.picklable_caltools import _initialize_global_data, _standalone_objective_function


class CalibrationLoggingUtility:
    """
    Utility class for managing logging, progress tracking, and file management
    across different calibration methods.
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize the logging utility.
        
        Args:
            output_dir: Base directory for outputs (if None, uses default from configs)
        """
        if output_dir is None:
            from utils.get_configs import get_paths
            path_obj = get_paths()
            self.base_output_dir = path_obj.fig_save_path
        else:
            self.base_output_dir = output_dir
    
    def create_timestamped_directory(self, prefix="calibration_run"):
        """
        Create a timestamped output directory.
        
        Args:
            prefix: Prefix for the directory name
            
        Returns:
            tuple: (run_dir, timestamp) where run_dir is the full path
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.base_output_dir, f"{prefix}_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir, timestamp
    
    def setup_logging_environment(self, run_dir, log_filename="optimization_log.txt", 
                                progress_filename="progress_history.json"):
        """
        Set up logging files and return configured functions.
        
        Args:
            run_dir: Directory to create log files in
            log_filename: Name of the log file
            progress_filename: Name of the progress file
            
        Returns:
            tuple: (log_message_func, log_file_path, progress_file_path)
        """
        log_file = os.path.join(run_dir, log_filename)
        progress_file = os.path.join(run_dir, progress_filename)
        
        def log_message(message, print_to_console=True):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] {message}"
            with open(log_file, 'a') as f:
                f.write(log_entry + '\n')
            if print_to_console:
                print(log_entry)
        
        return log_message, log_file, progress_file
    
    def initialize_progress_tracking(self):
        """
        Initialize the standard progress tracking data structure.
        
        Returns:
            dict: Initialized progress tracking dictionary
        """
        return {
            'runs': [],
            'overall_best_rmse': [],
            'overall_best_params': [],
            'run_summaries': []
        }
    
    def initialize_run_progress(self, run_number):
        """
        Initialize progress tracking for a specific run.
        
        Args:
            run_number: Current run number (1-indexed)
            
        Returns:
            dict: Initialized run progress dictionary
        """
        return {
            'run_number': run_number,
            'iterations': [],
            'best_rmse_history': [],
            'mean_rmse_history': [],
            'std_rmse_history': [],
            'convergence_info': {
                'converged': False,
                'iteration': None,
                'reason': 'Maximum iterations reached'
            }
        }
    
    def save_progress_data(self, progress_file, progress_data):
        """
        Save progress data to JSON file.
        
        Args:
            progress_file: Path to the progress file
            progress_data: Progress data dictionary to save
        """
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    def log_calibration_start(self, log_message_func, method_name, **kwargs):
        """
        Log the start of a calibration process with parameters.
        
        Args:
            log_message_func: Function to use for logging
            method_name: Name of the calibration method
            **kwargs: Key-value pairs of parameters to log
        """
        log_message_func(f"Starting {method_name} optimization", True)
        for key, value in kwargs.items():
            log_message_func(f"{key}: {value}", True)
    
    def log_run_start(self, log_message_func, run_number, total_runs):
        """
        Log the start of a specific run.
        
        Args:
            log_message_func: Function to use for logging
            run_number: Current run number (1-indexed)
            total_runs: Total number of runs
        """
        log_message_func(f"Starting run {run_number}/{total_runs}", True)
    
    def log_run_completion(self, log_message_func, run_number, total_runs, 
                          final_rmse, duration_seconds):
        """
        Log the completion of a specific run.
        
        Args:
            log_message_func: Function to use for logging
            run_number: Current run number (1-indexed)
            total_runs: Total number of runs
            final_rmse: Final RMSE achieved in this run
            duration_seconds: Duration of the run in seconds
        """
        log_message_func(f"Run {run_number} completed in {duration_seconds:.1f}s. "
                        f"Final RMSE: {final_rmse:.6f}", True)
    
    def log_overall_completion(self, log_message_func, total_duration, best_overall_rmse):
        """
        Log the completion of all runs.
        
        Args:
            log_message_func: Function to use for logging
            total_duration: Total duration of all runs in seconds
            best_overall_rmse: Best RMSE achieved across all runs
        """
        log_message_func(f"All runs completed in {total_duration:.1f}s", True)
        log_message_func(f"Best overall RMSE: {best_overall_rmse:.6f}", True)
    
    def log_new_best(self, log_message_func, run_number, iteration, new_rmse):
        """
        Log when a new best RMSE is found.
        
        Args:
            log_message_func: Function to use for logging
            run_number: Current run number (1-indexed)
            iteration: Current iteration number (1-indexed)
            new_rmse: New best RMSE value
        """
        log_message_func(f"Run {run_number}, Iter {iteration}: New best RMSE = {new_rmse:.6f}", False)
    
    def log_progress(self, log_message_func, run_number, iteration, total_iterations,
                    best_rmse, mean_rmse, no_improvement_count):
        """
        Log periodic progress updates.
        
        Args:
            log_message_func: Function to use for logging
            run_number: Current run number (1-indexed)
            iteration: Current iteration number (1-indexed)
            total_iterations: Total iterations for this run
            best_rmse: Current best RMSE
            mean_rmse: Current mean RMSE
            no_improvement_count: Count of iterations without improvement
        """
        log_message_func(f"Run {run_number}, Iter {iteration}/{total_iterations}: "
                        f"Best RMSE = {best_rmse:.6f}, "
                        f"Mean RMSE = {mean_rmse:.6f}, "
                        f"No improvement = {no_improvement_count}", False)
    
    def log_convergence(self, log_message_func, run_number, iteration, reason):
        """
        Log when convergence is reached.
        
        Args:
            log_message_func: Function to use for logging
            run_number: Current run number (1-indexed)
            iteration: Iteration where convergence occurred
            reason: Reason for convergence
        """
        log_message_func(f"Run {run_number} converged at iteration {iteration}: {reason}", True)
    
    def write_configuration_header(self, log_file, method_name, **config_params):
        """
        Write a configuration header to the log file for special calibration types.
        
        Args:
            log_file: Path to the log file
            method_name: Name of the calibration method
            **config_params: Key-value pairs of configuration parameters
        """
        with open(log_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"{method_name.upper()} CALIBRATION CONFIGURATION\n")
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output Directory: {os.path.dirname(log_file)}\n\n")
            
            # Write configuration parameters
            for section_name, section_params in config_params.items():
                if isinstance(section_params, dict):
                    f.write(f"{section_name.upper()}:\n")
                    f.write("-" * 40 + "\n")
                    for key, value in section_params.items():
                        f.write(f"{key}: {value}\n")
                    f.write("\n")
                else:
                    f.write(f"{section_name}: {section_params}\n")
            
            f.write("=" * 80 + "\n")
            f.write("OPTIMIZATION LOG STARTS BELOW\n")
            f.write("=" * 80 + "\n\n")
    
    def save_run_summary(self, progress_data, run_number, final_rmse, iterations_completed, 
                        duration_seconds, convergence_info):
        """
        Add a run summary to the progress data.
        
        Args:
            progress_data: Progress tracking dictionary
            run_number: Run number (1-indexed)
            final_rmse: Final RMSE for this run
            iterations_completed: Number of iterations completed
            duration_seconds: Duration of the run in seconds
            convergence_info: Dictionary with convergence details
        """
        progress_data['runs'].append(run_number)
        progress_data['run_summaries'].append({
            'run_number': run_number,
            'final_rmse': float(final_rmse),
            'iterations_completed': iterations_completed,
            'duration_seconds': duration_seconds,
            'convergence_info': convergence_info
        })
    
    def update_overall_best(self, progress_data, run_number, best_rmse, best_params):
        """
        Update the overall best results in progress data.
        
        Args:
            progress_data: Progress tracking dictionary
            run_number: Run number (1-indexed)
            best_rmse: Best RMSE achieved so far
            best_params: Best parameters achieved so far
        """
        progress_data['overall_best_rmse'].append(float(best_rmse))
        progress_data['overall_best_params'].append(
            best_params.tolist() if hasattr(best_params, 'tolist') else best_params
        )
    
    def create_optimization_summary(self, run_dir, method_name, best_params, best_rmse, 
                                  progress_data, total_duration):
        """
        Create a comprehensive optimization summary file.
        
        Args:
            run_dir: Directory to save the summary
            method_name: Name of the calibration method
            best_params: Best parameters achieved
            best_rmse: Best RMSE achieved
            progress_data: Progress tracking data
            total_duration: Total duration of all runs
        """
        summary_file = os.path.join(run_dir, f"{method_name.lower().replace(' ', '_')}_summary.txt")
        
        with open(summary_file, 'w') as f:
            f.write(f"{method_name.upper()} OPTIMIZATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Optimization completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total duration: {total_duration:.1f} seconds\n")
            f.write(f"Best overall RMSE: {best_rmse:.6f}\n")
            f.write(f"Number of runs: {len(progress_data['runs'])}\n\n")
            
            if hasattr(best_params, 'tolist'):
                f.write("Best parameters:\n")
                f.write("-" * 20 + "\n")
                for i, param_value in enumerate(best_params):
                    f.write(f"  Parameter {i+1}: {param_value:.6f}\n")
                f.write("\n")
            
            f.write("Run summaries:\n")
            f.write("-" * 20 + "\n")
            for summary in progress_data['run_summaries']:
                f.write(f"  Run {summary['run_number']}: RMSE={summary['final_rmse']:.6f}, "
                       f"Iterations={summary['iterations_completed']}, "
                       f"Duration={summary['duration_seconds']:.1f}s, "
                       f"Converged={'Yes' if summary['convergence_info']['converged'] else 'No'}\n")

    def create_twolayer_summary(self, run_dir, upper_layer_params, lower_layer_params, 
                               final_cost, total_duration, upper_run_dir):
        """
        Create a summary file specifically for two-layer nested calibration.
        
        Args:
            run_dir: Directory to save the summary
            upper_layer_params: Best upper layer parameters
            lower_layer_params: Best lower layer parameters
            final_cost: Final combined cost
            total_duration: Total duration of optimization
            upper_run_dir: Directory where upper layer results are stored
        """
        summary_file = os.path.join(run_dir, "twolayer_calibration_summary.txt")
        
        with open(summary_file, 'w') as f:
            f.write("TWO-LAYER NESTED CALIBRATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Calibration completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total duration: {total_duration:.1f} seconds\n")
            f.write(f"Final combined cost: {final_cost:.6f}\n\n")
            
            # f.write("UPPER LAYER PARAMETERS (System Parameters):\n")
            # f.write("-" * 40 + "\n")
            # if hasattr(upper_layer_params, 'tolist'):
            #     for i, param_value in enumerate(upper_layer_params):
            #         f.write(f"  Parameter {i+1}: {param_value:.6f}\n")
            # else:
            #     for i, param_value in enumerate(upper_layer_params):
            #         f.write(f"  Parameter {i+1}: {param_value:.6f}\n")
            # f.write("\n")
            
            # f.write("LOWER LAYER PARAMETERS (Gain Parameters):\n")
            # f.write("-" * 40 + "\n")
            # if hasattr(lower_layer_params, 'tolist'):
            #     for i, param_value in enumerate(lower_layer_params):
            #         f.write(f"  Parameter {i+1}: {param_value:.6f}\n")
            # else:
            #     for i, param_value in enumerate(lower_layer_params):
            #         f.write(f"  Parameter {i+1}: {param_value:.6f}\n")
            # f.write("\n")
            
            f.write(f"Upper layer results directory: {upper_run_dir}\n")
            f.write(f"Main results directory: {run_dir}\n")
    
    def log_upper_layer_progress(self, log_message_func, particle_number, total_particles, 
                                upper_params, lower_cost, iteration_number=None):
        """
        Log progress for upper layer optimization during two-layer nested calibration.
        
        Args:
            log_message_func: Function to use for logging
            particle_number: Current particle number (1-indexed)
            total_particles: Total number of particles in upper layer
            upper_params: Upper layer parameters for this particle
            lower_cost: Cost returned from lower layer optimization
            iteration_number: Optional iteration number if available
        """
        if iteration_number is not None:
            log_message_func(f"Upper Layer - Iter {iteration_number}, Particle {particle_number}/{total_particles}: "
                           f"Params={upper_params}, Lower Cost={lower_cost:.6f}", False)
        else:
            log_message_func(f"Upper Layer - Particle {particle_number}/{total_particles}: "
                           f"Params={upper_params}, Lower Cost={lower_cost:.6f}", False)
    
    def log_upper_layer_summary(self, log_message_func, best_upper_params, best_combined_cost, 
                               total_particles_evaluated, total_duration):
        """
        Log a summary of the upper layer optimization results.
        
        Args:
            log_message_func: Function to use for logging
            best_upper_params: Best upper layer parameters found
            best_combined_cost: Best combined cost achieved
            total_particles_evaluated: Total number of particles evaluated
            total_duration: Total duration of upper layer optimization
        """
        log_message_func("=" * 60, True)
        log_message_func("UPPER LAYER OPTIMIZATION SUMMARY", True)
        log_message_func("=" * 60, True)
        log_message_func(f"Total particles evaluated: {total_particles_evaluated}", True)
        log_message_func(f"Best combined cost: {best_combined_cost:.6f}", True)
        log_message_func(f"Best upper layer parameters: {best_upper_params}", True)
        log_message_func(f"Total duration: {total_duration:.1f} seconds", True)
        log_message_func("=" * 60, True)

    def create_simple_calibration_summary(self, run_dir, method_name, final_mse, 
                                        iterations_completed, total_duration, parameters=None):
        """
        Create a simple summary file for basic calibration methods.
        
        Args:
            run_dir: Directory to save the summary
            method_name: Name of the calibration method
            final_mse: Final MSE achieved
            iterations_completed: Number of iterations completed
            total_duration: Total duration of optimization
            parameters: Optional parameters to include in summary
        """
        summary_file = os.path.join(run_dir, f"{method_name.lower().replace(' ', '_')}_summary.txt")
        
        with open(summary_file, 'w') as f:
            f.write(f"{method_name.upper()} CALIBRATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Calibration completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total duration: {total_duration:.1f} seconds\n")
            f.write(f"Final MSE: {final_mse:.6f}\n")
            f.write(f"Iterations completed: {iterations_completed}\n\n")
            
            if parameters is not None:
                f.write("Parameters:\n")
                f.write("-" * 20 + "\n")
                if hasattr(parameters, 'tolist'):
                    for i, param_value in enumerate(parameters):
                        f.write(f"  Parameter {i+1}: {param_value:.6f}\n")
                else:
                    for i, param_value in enumerate(parameters):
                        f.write(f"  Parameter {i+1}: {param_value:.6f}\n")
                f.write("\n")


def get_perturbation_event_times(file_path, units='cents', epsilon=1e-10):
    df = pd.read_csv(file_path)

    time_col=0
    perturb_col=1

    # Get the series
    time = df.iloc[:, time_col]
    perturbation = df.iloc[:, perturb_col]

    # Find index of maximum absolute value (peak of perturbation)
    max_abs_val = abs(perturbation).max()
    print('max_abs_val', max_abs_val)

    # Find the actual value (could be negative) at the maximum absolute point
    max_idx = (perturbation[abs(perturbation) > (max_abs_val - epsilon)].index[0])
    max_val = perturbation.iloc[max_idx]
    print('max_idx', max_idx)

    if units == 'cents':
        # Find index of first non-zero perturbation using epsilon tolerance
        first_pert_idx = (perturbation[abs(perturbation) > epsilon].index[0]) - 1
        print('first_pert_idx', first_pert_idx)
        full_pert_idx = max_idx
        full_pert_val = max_val
    elif units == 'multiplier':
        if max_abs_val >= 1.0-epsilon:
            first_pert_idx = (perturbation[abs(perturbation) < (1-epsilon)].index[0]) - 1
            print('first_pert_idx', first_pert_idx)
            full_pert_val = abs(perturbation).min()
            full_pert_idx = (perturbation[abs(perturbation) < (full_pert_val + epsilon)].index[0]) - 1
        else:
            first_pert_idx = (perturbation[abs(perturbation) > (1-epsilon)].index[0]) - 1
            print('first_pert_idx', first_pert_idx)
            full_pert_idx = max_idx
            full_pert_val = max_val

    print('full_pert_val', full_pert_val)
    print('full_pert_idx', full_pert_idx)
    time_at_full_pert = time.iloc[full_pert_idx]

    # Get times
    time_at_first_pert = time.iloc[first_pert_idx]
    

    return time_at_first_pert, time_at_full_pert, full_pert_val
   

class PitchPertCalibrator:
    def __init__(self, params_obj, target_response, pert_signal, T_sim, truncate_start, truncate_end, sensor_processor=AbsoluteSensorProcessor()):
        """
        Initialize the calibrator with system parameters and data.
        
        Args:
            params_obj: Object containing system parameters
            target_response: Target response data to match
            pert_signal: Perturbation signal
            T_sim: Time series for simulation
            truncate_start: Start time for truncation
            truncate_end: End time for truncation
            sensor_processor: Sensor processor to use (default: AbsoluteSensorProcessor)
        """
        self.params_obj = params_obj
        self.target_response = target_response
        self.pert_signal = pert_signal
        self.T_sim = T_sim
        self.truncate_start = truncate_start
        self.truncate_end = truncate_end
        self.sensor_processor = sensor_processor
        self.mse_history = []
        
        # Initialize logging utility
        self.logging_utility = CalibrationLoggingUtility()
        
    def objective_function(self, params):
        """
        Objective function for parameter optimization.
        
        Args:
            params: 2D array of parameters where each row represents a particle's parameters
        
        Returns:
            Array of MSE values (one per particle) for PySwarms, or single MSE for other optimizers
        """
        print('params shape:', params.shape if hasattr(params, 'shape') else 'not numpy array')
        
        # Handle PySwarms vectorized evaluation (multiple particles)
        if isinstance(params, np.ndarray) and params.ndim == 2:
            print(f'PySwarms vectorized evaluation: {params.shape[0]} particles, {params.shape[1]} parameters each')
            costs = np.zeros(params.shape[0])
            
            for i in range(params.shape[0]):
                particle_params = params[i]
                print(f'Processing particle {i+1}/{params.shape[0]}')
                costs[i] = self._evaluate_single_particle(particle_params)
            
            return costs
        else:
            # Handle single particle evaluation (for other optimizers or debugging)
            print('Single particle evaluation')
            return self._evaluate_single_particle(params)
    
    def _evaluate_single_particle(self, params, null_values_spec=None):
        """
        Evaluate a single particle's parameters.
        
        Args:
            params: 1D array of parameters for a single particle
        
        Returns:
            MSE value for this particle
        """
        print('single particle params:', params)
        self._set_current_params(params, null_values_spec=null_values_spec)
        print('objective function current_params', self.current_params)
        
        if self.params_obj.system_type == 'DIVA':
            print('Diva sensor processor')
            system = DIVAController(self.sensor_processor, self.T_sim, self.params_obj.dt, self.pert_signal.signal, self.pert_signal.start_ramp_up, self.target_response, self.current_params)
            system.simulate(self.params_obj.kearney_name)
        else:
            # Template system - use existing logic
            # Create system with current parameters
            system = Controller(
                sensor_processor=self.sensor_processor, 
                params=self.current_params,
                ref_type=self.params_obj.ref_type, 
                dist_custom=self.pert_signal.signal,
                dist_type=['Auditory'],
                timeseries=self.T_sim
            )
            
            # Run simulation
            system.simulate_with_2sensors(
                delta_t_s_aud=int(self.current_params['sensor_delay_aud']),
                delta_t_s_som=int(self.current_params['sensor_delay_som']),
                delta_t_a=int(self.current_params['actuator_delay'])
            )
        
        # Calculate MSE between simulation and calibration data
        timeseries_truncated, system_response_truncated = truncate_data(
            self.T_sim, system.x, self.truncate_start, self.truncate_end
        )
        
        mse = system.mse(system_response_truncated, self.target_response, check_stability=True)
        print(f'Particle MSE: {mse}')
        return mse

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
        param_config, bounds, x0 = calibration_info_pack(self.params_obj, print_opt=['print'], custom_label='Calibrate', null_values=False)

        # Reset MSE history
        self.mse_history = []

        # Run optimization
        result = minimize(
            self.objective_function,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iterations},
            callback=self.callback_function
        )
        
        # Setup output directory using logging utility
        run_dir, timestamp = self.logging_utility.create_timestamped_directory(prefix="standard_calibration")
        
        # Setup logging using logging utility
        log_message, log_file, progress_file = self.logging_utility.setup_logging_environment(run_dir)
        
        # Log calibration completion
        log_message("Standard calibration completed", True)
        log_message(f"Final MSE: {result.fun}", True)
        log_message(f"Number of iterations: {result.nit}", True)
        log_message(f"Optimization success: {result.success}", True)
        log_message(f"Message: {result.message}", True)
        log_message(f"Number of MSE values recorded: {len(self.mse_history)}", True)
        
        # Create simple calibration summary
        self.logging_utility.create_simple_calibration_summary(
            run_dir, "Standard Calibration", result.fun, result.nit, 0, result.x
        )
        
        # Apply optimized parameters to the model
        self.apply_optimized_params(result.x, run_dir)
        
        # Create convergence plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.mse_history)
        plt.xlabel('Iteration')
        plt.ylabel('MSE')
        plt.title('Standard Calibration Convergence')
        plt.grid(True)
        plt.savefig(os.path.join(run_dir, 'convergence_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return self.params_obj, self.mse_history, run_dir
    
    def particle_swarm_calibrate(self, num_particles=10000, max_iters=1000, convergence_tol=0.01, runs=10, 
                                log_interval=10, save_interval=50, output_dir=None):
        """
        Enhanced particle swarm calibration with comprehensive logging and monitoring.
        
        Args:
            num_particles: Number of particles in the swarm
            max_iters: Maximum iterations per run
            convergence_tol: Convergence tolerance
            runs: Number of independent runs
            log_interval: How often to log progress (iterations)
            save_interval: How often to save intermediate results (iterations)
            output_dir: Directory to save outputs (if None, uses default)
            
        Returns:
            tuple: (optimized_params, mse_history, run_dir) where run_dir is the timestamped output directory
        """
        plt.use('Agg')  # Use non-interactive backend for SSH
        
        # Setup output directory using logging utility
        run_dir, timestamp = self.logging_utility.create_timestamped_directory(prefix="particle_swarm_run")
        
        # Setup logging using logging utility
        log_message, log_file, progress_file = self.logging_utility.setup_logging_environment(run_dir)
        
        # Initialize tracking variables 
        param_config, bounds, x0, current_params = calibration_info_pack(self.params_obj, print_opt=['print'], custom_label='Particle Swarm', null_values=False)
        bounds = np.array(bounds)  # Convert to numpy array for indexing
        print('bounds', bounds)

        num_params = len(bounds)
        best_overall_rmse = np.inf
        best_overall_params = None
        
        # Progress tracking using logging utility
        progress_data = self.logging_utility.initialize_progress_tracking()
        
        # Log calibration start using logging utility
        self.logging_utility.log_calibration_start(
            log_message, 
            "particle swarm", 
            particles=num_particles, 
            max_iters=max_iters, 
            convergence_tol=convergence_tol,
            output_directory=run_dir
        )
        
        start_time = datetime.now()
        
        for run in range(runs):
            run_start_time = datetime.now()
            self.logging_utility.log_run_start(log_message, run + 1, runs)
            
            #particles = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_particles, num_params))
            particles = self.quantized_uniform(bounds[:, 0], bounds[:, 1], precision=3, size=(num_particles, num_params))
            best_rmse = np.inf
            best_params = None
            best_history = []
            no_improvement_count = 0
            
            # Run-specific progress tracking using logging utility
            run_progress = self.logging_utility.initialize_run_progress(run + 1)
            
            for it in range(max_iters):
                # #Scatter plot for test parameter
                # plt.scatter(particles[:, 0], particles[:, 1])
                # plt.xlabel('Parameter 1')
                # plt.ylabel('Parameter 2')
                # plt.title(f'Run {run+1}, Iter {it+1}')
                # #Save plot
                # plt.savefig(os.path.join(run_dir, f'run_{run+1}_iter_{it+1}_particles.png'), dpi=300, bbox_inches='tight')
                # plt.close()
                # print('Saved plot to', os.path.join(run_dir, f'run_{run+1}_iter_{it+1}_particles.png'))
                
                # Evaluate all particles
                rmses = np.array([self.objective_function(p) for p in particles])
                current_best_rmse = np.min(rmses)
                current_best_idx = np.argmin(rmses)
                current_best_params = particles[current_best_idx]
                
                # Track progress
                best_history.append(current_best_rmse)
                run_progress['iterations'].append(it + 1)
                run_progress['best_rmse_history'].append(float(current_best_rmse))
                run_progress['mean_rmse_history'].append(float(np.mean(rmses)))
                run_progress['std_rmse_history'].append(float(np.std(rmses)))
                
                # Check for improvement
                if current_best_rmse < best_rmse - 1e-8:
                    best_rmse = current_best_rmse
                    best_params = current_best_params.copy()
                    no_improvement_count = 0
                    self.logging_utility.log_new_best(log_message, run + 1, it + 1, best_rmse)
                else:
                    no_improvement_count += 1
                
                # Log progress periodically
                if (it + 1) % log_interval == 0:
                    self.logging_utility.log_progress(
                        log_message, run + 1, it + 1, max_iters, 
                        best_rmse, np.mean(rmses), no_improvement_count
                    )
                
                # Save intermediate results periodically
                if (it + 1) % save_interval == 0:
                    self._save_intermediate_results(run_dir, run, it, best_params, best_rmse, 
                                                  particles, rmses, progress_data)
                
                # Check convergence
                convergence_reached = False
                convergence_reason = ""
                if np.max(rmses) - np.min(rmses) < convergence_tol * best_rmse:
                    convergence_reached = True
                    convergence_reason = "Tolerance reached"
                elif no_improvement_count >= 100:
                    convergence_reached = True
                    convergence_reason = "No improvement for 100 iterations"
                
                if convergence_reached:
                    self.logging_utility.log_convergence(log_message, run + 1, it + 1, convergence_reason)
                    run_progress['convergence_info'] = {
                        'converged': True,
                        'iteration': it + 1,
                        'reason': convergence_reason
                    }
                    break
                
                # Paper's method: Replace fraction with random linear combinations of best fits
                elite_fraction = 0.1
                replacement_fraction = 0.3

                elite_size = max(1, int(num_particles * elite_fraction))
                replacement_size = int(num_particles * replacement_fraction)
                regeneration_size = num_particles - elite_size - replacement_size

                # Identify elite particles (best fits)
                elite_idx = np.argsort(rmses)[:elite_size]
                elite = particles[elite_idx]

                # Generate new particles
                new_particles = []

                # Keep elite particles
                new_particles.extend(elite)

                # Generate crossover particles
                for _ in range(replacement_size):
                    parent1 = elite[np.random.randint(elite_size)]
                    parent2 = elite[np.random.randint(elite_size)]
                    weight = np.random.uniform(0, 1)
                    new_particle = weight * parent1 + (1 - weight) * parent2
                    
                    # Clip and quantize
                    new_particle = np.clip(new_particle, bounds[:, 0], bounds[:, 1])
                    steps = 10 ** 3
                    quantized_particle = np.round(new_particle * steps) / steps
                    quantized_particle = np.clip(quantized_particle, bounds[:, 0], bounds[:, 1])
                    
                    new_particles.append(quantized_particle)

                # Generate random particles for the remainder
                if regeneration_size > 0:
                    new_particles.extend(self.quantized_uniform(bounds[:, 0], bounds[:, 1], precision=3, size=(regeneration_size, num_params)))

                # Update particle swarm
                particles = np.array(new_particles)
            
            # Run completed
            run_end_time = datetime.now()
            run_duration = (run_end_time - run_start_time).total_seconds()
            
            self.logging_utility.log_run_completion(log_message, run + 1, runs, best_rmse, run_duration)
            
            # Update overall best if this run was better
            if best_rmse < best_overall_rmse:
                best_overall_rmse = best_rmse
                best_overall_params = best_params.copy()
                log_message(f"New overall best RMSE: {best_overall_rmse:.6f}", True)
            
            # Store run summary using logging utility
            self.logging_utility.save_run_summary(
                progress_data, run + 1, best_rmse, len(best_history), 
                run_duration, run_progress['convergence_info']
            )
            
            # Update overall best using logging utility
            self.logging_utility.update_overall_best(
                progress_data, run + 1, best_overall_rmse, best_overall_params
            )
            
            # Save run progress using logging utility
            self.logging_utility.save_progress_data(progress_file, progress_data)
            
            # Create run plots
            self._create_run_plots(run_dir, run, run_progress)
        
        # All runs completed
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        self.logging_utility.log_overall_completion(log_message, total_duration, best_overall_rmse)
        
        # Apply the best parameters to the model
        if best_overall_params is not None:
            self.apply_optimized_params(best_overall_params, run_dir)
        
        # Create optimization summary using logging utility
        self.logging_utility.create_optimization_summary(
            run_dir, "particle swarm", best_overall_params, best_overall_rmse, 
            progress_data, total_duration
        )
        
        # Save final results
        self._save_final_results(run_dir, best_overall_params, best_overall_rmse, progress_data)
        
        # Return the optimized parameters, mse history, and run directory
        return self.params_obj, best_overall_rmse, run_dir

    def pyswarms_calibrate(self, num_particles=10000, max_iters=1000, convergence_tol=0.01, runs=10, 
                            log_interval=1, save_interval=50, parallel_opt=False, output_dir=None, custom_objective=None, null_values=None):
        """
        PySwarms calibration with comprehensive logging and monitoring.
        
        Args:
            num_particles: Number of particles in the swarm
            max_iters: Maximum iterations per run
            convergence_tol: Convergence tolerance
            runs: Number of independent runs
            log_interval: How often to log progress (iterations)
            save_interval: How often to save intermediate results (iterations)
            output_dir: Directory to save outputs (if None, uses default)
            
        Returns:
            tuple: (optimized_params, mse_history, run_dir) where run_dir is the timestamped output directory
        """
    
        # Setup output directory using logging utility
        if output_dir is None:
            run_dir, timestamp = self.logging_utility.create_timestamped_directory(prefix="pyswarms_run")
        else:
            run_dir = output_dir
        print('run_dir', run_dir)
        # Setup logging using logging utility
        log_message, log_file, progress_file = self.logging_utility.setup_logging_environment(run_dir)

        # Initialize global data for multiprocessing
        _initialize_global_data(
            self.params_obj,
            self.target_response,
            self.pert_signal,
            self.T_sim,
            self.truncate_start,
            self.truncate_end,
            self.sensor_processor
        )

        # Initialize tracking variables 
        param_config, bounds, x0, current_params = calibration_info_pack(self.params_obj, print_opt=['print'], custom_label='PySwarms', null_values=null_values)
        bounds = np.array(bounds)  # Convert to numpy array for indexing
        print('bounds', bounds)
        
        # Convert bounds format for PySwarms: (lower_bounds, upper_bounds)
        lower_bounds = bounds[:, 0]  # First column contains lower bounds
        upper_bounds = bounds[:, 1]  # Second column contains upper bounds
        pyswarms_bounds = (lower_bounds, upper_bounds)
        print('pyswarms_bounds format:', pyswarms_bounds)
        
        num_params = len(bounds)
        best_overall_rmse = np.inf
        best_overall_params = None
        
        # Progress tracking using logging utility
        progress_data = self.logging_utility.initialize_progress_tracking()
        
        # Log calibration start using logging utility
        self.logging_utility.log_calibration_start(
            log_message, 
            "PySwarms", 
            particles=num_particles, 
            max_iters=max_iters, 
            convergence_tol=convergence_tol,
            output_directory=run_dir
        )

        log_message("Calibration Settings:", True)
        for key, value in self.params_obj.cal_set_dict.items():
            log_message(f"{key}: {value}", True)
        
        start_time = datetime.now()
        
        for run in range(runs):
            run_start_time = datetime.now()
            self.logging_utility.log_run_start(log_message, run + 1, runs)

            # PySwarms options dictionary
            options = {
                'c1': self.params_obj.cal_set_dict['c1'],  # cognitive parameter
                'c2': self.params_obj.cal_set_dict['c2'],  # social parameter
                'w': self.params_obj.cal_set_dict['w'],   # inertia weight
                'k': self.params_obj.cal_set_dict['k'],     # number of neighbors for topology
                'p': self.params_obj.cal_set_dict['p']      # p-norm for distance calculation
            }
            
            # Initialize PySwarms optimizer
            optimizer = ps.single.GlobalBestPSO(
                n_particles=num_particles,
                dimensions=num_params,
                options=options,
                bounds=pyswarms_bounds
            )

            optimizer.ftol = 1e-6
            optimizer.ftol_iter = 5
            
            # Run-specific progress tracking using logging utility
            run_progress = self.logging_utility.initialize_run_progress(run + 1)
            
            # Progress tracking will be done after optimization completes
            # since PySwarms doesn't support callbacks in the same way
            
            # Run optimization
            set_objective_function = custom_objective if custom_objective is not None else _standalone_objective_function


            best_cost, best_pos = optimizer.optimize(
                objective_func=set_objective_function,
                n_processes=psutil.cpu_count(logical=False)-2 if parallel_opt else None,
                iters=max_iters,
                verbose=False
            )

            print(run, 'best_cost', best_cost)
            print(run, 'best_pos', best_pos)
            # Extract progress data from optimizer
            if hasattr(optimizer, 'cost_history') and optimizer.cost_history:
                for i, costs in enumerate(optimizer.cost_history):
                    iteration = i + 1
                    best_cost_iter = np.min(costs)
                    mean_cost_iter = np.mean(costs)
                    std_cost_iter = np.std(costs)
                    
                    # Track progress
                    run_progress['iterations'].append(iteration)
                    run_progress['best_rmse_history'].append(float(best_cost_iter))
                    run_progress['mean_rmse_history'].append(float(mean_cost_iter))
                    run_progress['std_rmse_history'].append(float(std_cost_iter))
                    
                    # Log progress periodically
                    if iteration % log_interval == 0:
                        log_message(f"Run {run + 1}, Iter {iteration}/{max_iters}: Best RMSE = {best_cost_iter:.6f}, "
                                f"Mean RMSE = {mean_cost_iter:.6f}", False)
                    
                    # Save intermediate results periodically
                    if iteration % save_interval == 0:
                        print(run, 'iteration', iteration)
                        print(run, 'optimizer.best_pos', optimizer.best_pos)
                        print(run, 'best_cost_iter', best_cost_iter)
                        print(run, 'optimizer.pos', optimizer.pos)
                        print(run, 'costs', costs)
                        self._save_intermediate_results(run_dir, run, iteration-1, optimizer.best_pos, best_cost_iter, 
                                                    optimizer.pos, costs, progress_data)
            
            # Run completed
            run_end_time = datetime.now()
            run_duration = (run_end_time - run_start_time).total_seconds()
            
            self.logging_utility.log_run_completion(log_message, run + 1, runs, best_cost, run_duration)
            
            # Update overall best if this run was better
            if best_cost < best_overall_rmse:
                best_overall_rmse = best_cost
                best_overall_params = best_pos.copy()
                log_message(f"New overall best RMSE: {best_overall_rmse:.6f}", True)
            
            # Store run summary using logging utility
            self.logging_utility.save_run_summary(
                progress_data, run + 1, best_cost, len(run_progress['iterations']), 
                run_duration, run_progress['convergence_info']
            )
            
            # Update overall best using logging utility
            self.logging_utility.update_overall_best(
                progress_data, run + 1, best_overall_rmse, best_overall_params
            )
            
            # Save run progress
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
            
            # Create run plots
            self._create_run_plots(run_dir, run, run_progress)
        
        # All runs completed
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        self.logging_utility.log_overall_completion(log_message, total_duration, best_overall_rmse)
        
        # Apply the best parameters to the model
        if best_overall_params is not None:
            self.apply_optimized_params(best_overall_params, run_dir)
        
        # Create optimization summary using logging utility
        self.logging_utility.create_optimization_summary(
            run_dir, "PySwarms", best_overall_params, best_overall_rmse, 
            progress_data, total_duration
        )
        
        # Save final results
        self._save_final_results(run_dir, best_overall_params, best_overall_rmse, progress_data)
        
        return self.params_obj, best_overall_rmse, run_dir, best_overall_params


    def pyswarms_twolayer_calibrate(self, upper_particles=100, upper_iters=50, upper_runs=3,
                                    lower_particles=300, lower_iters=30, lower_runs=2,
                                    convergence_tol=0.01, log_interval=1, save_interval=25, output_dir=None):
        """
        Feasible nested calibration: small upper layer, larger lower layer.
        
        Args:
            upper_particles: Number of particles for system parameters (keep small: 50-200)
            upper_iters: Iterations for upper layer (keep small: 30-100)
            upper_runs: Runs for upper layer (keep small: 2-5)
            lower_particles: Number of particles for gain parameters (can be larger: 200-500)
            lower_iters: Iterations for lower layer (keep small: 20-50)
            lower_runs: Runs for lower layer (keep small: 1-3)
            convergence_tol: Convergence tolerance
            log_interval: How often to log progress
            save_interval: How often to save intermediate results
            output_dir: Directory to save outputs
        """
        # Define parameter layers
        upper_layer_params = ['A', 'B', 'C_aud', 'C_som', 'sensor_delay_aud', 'sensor_delay_som', 'actuator_delay']
        lower_layer_params = ['K_aud', 'K_som', 'L_aud', 'L_som']
        
        # Setup output directory using logging utility
        
        run_dir, timestamp = self.logging_utility.create_timestamped_directory(prefix="twolayer_run")
        
        print('run_dir', run_dir)
        # Setup logging using logging utility with custom filename
        log_message, log_file, progress_file = self.logging_utility.setup_logging_environment(
            run_dir, log_filename="twolayer_optimization_log.txt"
        )
        
        # Write initial configuration using logging utility
        upper_layer_config = {
            'Parameters': upper_layer_params,
            'Particles': upper_particles,
            'Iterations': upper_iters,
            'Runs': upper_runs
        }
        
        lower_layer_config = {
            'Parameters': lower_layer_params,
            'Particles': lower_particles,
            'Iterations': lower_iters,
            'Runs': lower_runs
        }
        
        calibration_settings = {}
        if hasattr(self.params_obj, 'cal_set_dict') and self.params_obj.cal_set_dict:
            calibration_settings = self.params_obj.cal_set_dict
        
        self.logging_utility.write_configuration_header(
            log_file,
            "two-layer nested calibration",
            upper_layer=upper_layer_config,
            lower_layer=lower_layer_config,
            calibration_settings=calibration_settings
        )
        
        # Log calibration start using logging utility
        self.logging_utility.log_calibration_start(
            log_message, 
            "two-layer nested", 
            upper_particles=upper_particles, 
            upper_iters=upper_iters, 
            upper_runs=upper_runs,
            lower_particles=lower_particles,
            lower_iters=lower_iters,
            lower_runs=lower_runs,
            convergence_tol=convergence_tol,
            output_directory=run_dir
        )
        
        # Store lower layer parameters for the nested objective function
        self._lower_layer_config = {
            'particles': lower_particles,
            'iterations': lower_iters,
            'runs': lower_runs,
            'convergence_tol': convergence_tol,
            'log_interval': log_interval,
            'save_interval': save_interval
        }
        
        # Pass the logging utility and run directory to the upper layer objective
        self._upper_layer_logging_utility = self.logging_utility
        self._upper_layer_run_dir = run_dir
        self._upper_layer_log_message = log_message
        self._upper_particles = upper_particles
        
        # Initialize tracking for upper layer optimization
        self._upper_layer_call_count = 0
        self._upper_layer_best_cost = np.inf
        self._upper_layer_best_params = None
        
        # Initialize global data for multiprocessing BEFORE creating the calibrator
        
        _initialize_global_data(
            self.params_obj,
            self.target_response,
            self.pert_signal,
            self.T_sim,
            self.truncate_start,
            self.truncate_end,
            self.sensor_processor
        )
        
        # Create upper layer calibrator
        upper_calibrator = PitchPertCalibrator(
            params_obj=self.params_obj,
            target_response=self.target_response,
            pert_signal=self.pert_signal,
            T_sim=self.T_sim,
            truncate_start=self.truncate_start,
            truncate_end=self.truncate_end,
            sensor_processor=self.sensor_processor
        )
            
        # Run upper layer optimization
        log_message("Starting upper layer optimization (system parameters)", True)
        start_time = datetime.now()
        
        cal_params, mse_history, upper_run_dir, best_params = upper_calibrator.pyswarms_calibrate(
            num_particles=upper_particles,
            max_iters=upper_iters,
            convergence_tol=convergence_tol,
            runs=upper_runs,
            log_interval=log_interval,
            save_interval=save_interval,
            parallel_opt=False,
            output_dir=run_dir,
            custom_objective=self.upper_layer_objective,
            null_values='upper layer'
        )
        
        upper_duration = (datetime.now() - start_time).total_seconds()
        log_message(f"Upper layer completed in {upper_duration:.1f}s", True)
        #log_message(f"Best upper layer parameters: {}", True)

        # Log final upper layer summary
        if hasattr(self, '_upper_layer_call_count') and hasattr(self, '_upper_layer_best_cost'):
            self.logging_utility.log_upper_layer_summary(
                log_message,
                cal_params,
                mse_history,
                self._upper_layer_call_count,
                upper_duration
            )
        
        # Create two-layer summary using logging utility
        self.logging_utility.create_twolayer_summary(
            run_dir, cal_params, None, mse_history, upper_duration, upper_run_dir
        )
        
        return self.params_obj, mse_history, run_dir, best_params

    def upper_layer_objective(self, params):
        """
        Objective function for upper layer that runs lower layer optimization.
        This is called for each particle in the upper layer swarm.
        """
        
        
        # Handle PySwarms vectorized evaluation (multiple particles)
        if isinstance(params, np.ndarray) and params.ndim == 2:
            # Vectorized evaluation: params is a 2D array of shape (n_particles, n_params)
            # We need to evaluate each particle separately
            costs = np.zeros(params.shape[0])
            for i in range(params.shape[0]):
                particle_params = params[i]
                costs[i] = self._upper_layer_evaluate_single_particle(particle_params)
                print(f'particle {i}, cost {costs[i]}')
            return costs

    def _upper_layer_evaluate_single_particle(self, particle_params):
        # Increment call counter and track progress
        if hasattr(self, '_upper_layer_call_count'):
            self._upper_layer_call_count += 1
            self.call_num = self._upper_layer_call_count
        else:
            self.call_num = "unknown"
        
        # Use the logging utility if available
        if hasattr(self, '_upper_layer_log_message'):
            self.log_message = self._upper_layer_log_message
            self.log_message(f"Upper layer objective call #{self.call_num} with params: {particle_params.shape}", False)
        else:
            # Fallback logging if no utility available
            print(f"Upper layer objective call #{self.call_num} with params: {particle_params.shape}")

        # Initialize global data for this evaluation
        _initialize_global_data(
            self.params_obj,
            self.target_response,
            self.pert_signal,
            self.T_sim,
            self.truncate_start,
            self.truncate_end,
            self.sensor_processor
        )
        
        # Create lower layer calibrator
        lower_calibrator = PitchPertCalibrator(
            params_obj=self.params_obj,
            target_response=self.target_response,
            pert_signal=self.pert_signal,
            T_sim=self.T_sim,
            truncate_start=self.truncate_start,
            truncate_end=self.truncate_end,
            sensor_processor=self.sensor_processor
        )
        
        # Run lower layer optimization for these upper layer parameters
        try:
            if hasattr(self, '_upper_layer_log_message'):
                self.log_message(f"Call #{self.call_num}: Starting lower layer optimization for upper params: {particle_params.shape}", False)
                self.log_message(f"Call #{self.call_num}: Lower layer config: particles={self._lower_layer_config['particles']}, "
                        f"iterations={self._lower_layer_config['iterations']}, "
                        f"runs={self._lower_layer_config['runs']}", False)
            
            cal_params, mse_history, lower_run_dir, best_params = lower_calibrator.pyswarms_calibrate(
                num_particles=self._lower_layer_config['particles'],
                max_iters=self._lower_layer_config['iterations'],
                convergence_tol=self._lower_layer_config['convergence_tol'],
                runs=self._lower_layer_config['runs'],
                log_interval=self._lower_layer_config['log_interval'],
                save_interval=self._lower_layer_config['save_interval'],
                parallel_opt=True,
                output_dir=self._upper_layer_run_dir,
                null_values='lower layer'
            )
            
            if hasattr(self, '_upper_layer_log_message'):
                self.log_message(f"Call #{self.call_num}: Lower layer optimization completed successfully", False)
                self.log_message(f"Call #{self.call_num}: Lower layer best params: {cal_params}", False)
                self.log_message(f"Call #{self.call_num}: Lower YOOHOO! layer best RMSE: {mse_history}", False)
            
            # Evaluate the combined system with both upper and lower layer parameters
            cost = mse_history  # Use the best RMSE from lower layer
            print('pre-dict')
            best_params_dict = self._get_parameter_dict(best_params)
            print('post-dict')
            if hasattr(self, '_upper_layer_log_message'):
                self.log_message(f"Call #{self.call_num}: Combined system evaluation completed. Total cost: {cost:.6f}", False)
                
            # Log progress summary
            if hasattr(self, '_upper_layer_call_count'):
                self.logging_utility.log_upper_layer_progress(
                    self.log_message, self.call_num, self._upper_particles, best_params_dict, cost
                )
            print(f'particle {self.call_num}, cost {cost}')
            return cost
            
        except Exception as e:
            error_msg = f"Call #{self.call_num}: Error in lower layer optimization: {e}"
            if hasattr(self, '_upper_layer_log_message'):
                self.log_message(error_msg, False)
            else:
                print(error_msg)
            return 1e10  # Return high cost for failed evaluations

    
    def eval_only(self, params):
        """
        Evaluate the objective function only.
        """
        return self._evaluate_single_particle(params)
            

    def quantized_uniform(self, low, high, precision, size=None):
        """
        Generate random numbers between `low` and `high` with specified decimal `precision`.
        This avoids post-hoc rounding and ensures evenly spaced values.
        
        Args:
            low (float or array): Lower bound.
            high (float or array): Upper bound.
            precision (int): Number of decimal places.
            size (int or tuple): Shape of the output array.
            
        Returns:
            np.ndarray: Array of quantized random values.
        """
        steps = 10 ** precision
        low_int = np.round(low * steps).astype(int)
        high_int = np.round(high * steps).astype(int)
        
        # Random integers in quantized grid
        rand_ints = np.random.randint(low_int, high_int + 1, size=size)
        return rand_ints.astype(float) / steps
        
    def _get_parameter_dict(self, best_params):
        """Convert parameter array to dictionary based on system type."""
        if self.params_obj.system_type == 'DIVA':
            config = get_params_for_implementation(self.params_obj.system_type, kearney_name=self.params_obj.kearney_name)
        elif self.params_obj.system_type == 'Template':
            print('dict-related error')
            config = get_params_for_implementation(self.params_obj.system_type, arb_name=self.params_obj.arb_name)
        return {param_name: value for param_name, value in zip(config, best_params)}
    
    def _save_intermediate_results(self, run_dir, run, iteration, best_params, best_rmse, particles, rmses, progress_data):
        """Save intermediate results during optimization."""
        import os
        import json
        
        # Save current best parameters
        if best_params is not None:
            param_file = os.path.join(run_dir, f"run_{run+1}_iter_{iteration+1}_best_params.json")
            param_data = {
                'run': run + 1,
                'iteration': iteration + 1,
                'best_rmse': float(best_rmse),
                'parameters': self._get_parameter_dict(best_params)
            }
            with open(param_file, 'w') as f:
                json.dump(param_data, f, indent=2)
        
        # Save swarm statistics
        stats_file = os.path.join(run_dir, f"run_{run+1}_iter_{iteration+1}_swarm_stats.json")
        stats_data = {
            'run': run + 1,
            'iteration': iteration + 1,
            'best_rmse': float(np.min(rmses)),
            'mean_rmse': float(np.mean(rmses)),
            'std_rmse': float(np.std(rmses)),
            'min_rmse': float(np.min(rmses)),
            'max_rmse': float(np.max(rmses))
        }
        with open(stats_file, 'w') as f:
            json.dump(stats_data, f, indent=2)
    
    def _create_run_plots(self, run_dir, run, run_progress):
        """Create plots for a specific run."""
        import matplotlib.pyplot as plt
        import os
        
        # Convergence plot
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(run_progress['iterations'], run_progress['best_rmse_history'], 'b-', linewidth=2, label='Best RMSE')
        plt.xlabel('Iteration')
        plt.ylabel('RMSE')
        plt.title(f'Run {run + 1}: Best RMSE Convergence')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(run_progress['iterations'], run_progress['mean_rmse_history'], 'r-', linewidth=2, label='Mean RMSE')
        plt.fill_between(run_progress['iterations'], 
                        np.array(run_progress['mean_rmse_history']) - np.array(run_progress['std_rmse_history']),
                        np.array(run_progress['mean_rmse_history']) + np.array(run_progress['std_rmse_history']),
                        alpha=0.3, color='red')
        plt.xlabel('Iteration')
        plt.ylabel('RMSE')
        plt.title(f'Run {run + 1}: Swarm Statistics')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(run_progress['iterations'], run_progress['std_rmse_history'], 'g-', linewidth=2, label='Std RMSE')
        plt.xlabel('Iteration')
        plt.ylabel('RMSE Std')
        plt.title(f'Run {run + 1}: Swarm Diversity')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.semilogy(run_progress['iterations'], run_progress['best_rmse_history'], 'b-', linewidth=2, label='Best RMSE (log scale)')
        plt.xlabel('Iteration')
        plt.ylabel('RMSE (log scale)')
        plt.title(f'Run {run + 1}: Convergence (Log Scale)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f'run_{run+1}_convergence.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_final_results(self, run_dir, best_params, best_rmse, progress_data):
        """Save final optimization results."""
        import os
        import json
        from datetime import datetime
        
        # Save final parameters
        final_params_file = os.path.join(run_dir, "final_optimized_params.json")
        param_data = {
            'optimization_completed': datetime.now().isoformat(),
            'best_rmse': float(best_rmse),
            'parameters': self._get_parameter_dict(best_params)
        }
        with open(final_params_file, 'w') as f:
            json.dump(param_data, f, indent=2)
        
        # Create overall convergence plot
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 10))
        
        # Plot best RMSE for each run
        plt.subplot(2, 3, 1)
        plt.plot(progress_data['runs'], progress_data['overall_best_rmse'], 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Run Number')
        plt.ylabel('Best RMSE')
        plt.title('Best RMSE by Run')
        plt.grid(True)
        
        # Plot run summaries
        plt.subplot(2, 3, 2)
        run_numbers = [summary['run_number'] for summary in progress_data['run_summaries']]
        final_rmses = [summary['final_rmse'] for summary in progress_data['run_summaries']]
        plt.bar(run_numbers, final_rmses, alpha=0.7)
        plt.xlabel('Run Number')
        plt.ylabel('Final RMSE')
        plt.title('Final RMSE by Run')
        plt.grid(True)
        
        # Plot run durations
        plt.subplot(2, 3, 3)
        durations = [summary['duration_seconds'] for summary in progress_data['run_summaries']]
        plt.bar(run_numbers, durations, alpha=0.7, color='green')
        plt.xlabel('Run Number')
        plt.ylabel('Duration (seconds)')
        plt.title('Run Duration')
        plt.grid(True)
        
        # Plot iterations completed
        plt.subplot(2, 3, 4)
        iterations = [summary['iterations_completed'] for summary in progress_data['run_summaries']]
        plt.bar(run_numbers, iterations, alpha=0.7, color='orange')
        plt.xlabel('Run Number')
        plt.ylabel('Iterations Completed')
        plt.title('Iterations per Run')
        plt.grid(True)
        
        # Plot convergence status
        plt.subplot(2, 3, 5)
        converged = [1 if summary['convergence_info']['converged'] else 0 for summary in progress_data['run_summaries']]
        plt.bar(run_numbers, converged, alpha=0.7, color='red')
        plt.xlabel('Run Number')
        plt.ylabel('Converged (1=Yes, 0=No)')
        plt.title('Convergence Status')
        plt.grid(True)
        plt.ylim(0, 1.2)
        
        # Overall progress
        plt.subplot(2, 3, 6)
        plt.plot(progress_data['runs'], progress_data['overall_best_rmse'], 'ro-', linewidth=3, markersize=10)
        plt.xlabel('Run Number')
        plt.ylabel('Overall Best RMSE')
        plt.title('Overall Progress')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, 'overall_optimization_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save summary text file
        summary_file = os.path.join(run_dir, "optimization_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("PARTICLE SWARM OPTIMIZATION SUMMARY\n")
            f.write("===================================\n\n")
            f.write(f"Optimization completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Best overall RMSE: {best_rmse:.6f}\n\n")
            f.write("Best parameters:\n")
            param_dict = self._get_parameter_dict(best_params)
            for param_name, param_value in param_dict.items():
                f.write(f"  {param_name}: {param_value:.6f}\n")
            f.write("\n")
            
            f.write("Run summaries:\n")
            for summary in progress_data['run_summaries']:
                f.write(f"  Run {summary['run_number']}: RMSE={summary['final_rmse']:.6f}, "
                       f"Iterations={summary['iterations_completed']}, "
                       f"Duration={summary['duration_seconds']:.1f}s, "
                       f"Converged={'Yes' if summary['convergence_info']['converged'] else 'No'}\n")

    def apply_optimized_params(self, optimized_params, run_dir):
        """
        Apply optimized parameters to the model.
        
        Args:
            optimized_params: Array of optimized parameter values
        """
        print('START: apply_optimized_params')
        readout_optimized_params(self.params_obj, output_dir=run_dir)
        if optimized_params is not None:
            if self.params_obj.system_type == 'DIVA':
                config = get_params_for_implementation(self.params_obj.system_type, kearney_name=self.params_obj.kearney_name)
            elif self.params_obj.system_type == 'Template':
                config = get_params_for_implementation(self.params_obj.system_type, arb_name=self.params_obj.arb_name)
            
            # Make optimized_params a dictionary
            optimized_params_dict = {
                param_name: value
                for param_name, value in zip(config, optimized_params)
            }

            for param_name in config:  
                if hasattr(self.params_obj, param_name):
                    setattr(self.params_obj, param_name, optimized_params_dict[param_name])
        readout_optimized_params(self.params_obj, output_dir=run_dir)
        calibration_info_pack(self.params_obj, print_opt=['print'], custom_label='After Applying Optimized Params')
        print('- END: apply_optimized_params')

    def _set_current_params(self, params, null_values_spec=None):
        """Set current_params dict to params"""
        print('params_obj type', type(self.params_obj))

        if null_values_spec is not None:
            null_values = null_values_spec
        else:
            null_values = self.params_obj.cal_set_dict['null_values']

        if self.params_obj.system_type == 'DIVA':
                config = get_params_for_implementation(self.params_obj.system_type, kearney_name=self.params_obj.kearney_name)
        elif self.params_obj.system_type == 'Template':
            print('self.params_obj.arb_name', self.params_obj.arb_name)
            config = get_params_for_implementation(self.params_obj.system_type, arb_name=self.params_obj.arb_name)
        else:
            print('WARNING: Unlisted system type')
        
        # Handle PySwarms parameter format - extract first row if 2D array
        if isinstance(params, np.ndarray) and params.ndim > 1:
            print(f'PySwarms detected: params shape {params.shape}, extracting first row')
            params = params[0]  # Extract first row for single particle evaluation
        
        if type(params) != type(self.params_obj):
            print('params is not the same type as params_obj')
            # Create a dictionary with individual parameter values
            param_dict = {}
            for i, param_name in enumerate(config):
                if i < len(params):
                    param_dict[param_name] = params[i]
                else:
                    print(f'Warning: Parameter {param_name} at index {i} not found in params array of length {len(params)}')
            
            temp_params = BlankParamsObject(**param_dict)
        else:
            temp_params = params
            
        current_params= get_current_params(self.params_obj, config, cal_only=True, null_values=null_values, params=temp_params)
        
        self.current_params = current_params
        
    def _apply_params_to_model(self, best_params):
        # Implementation of _apply_params_to_model method
        pass

