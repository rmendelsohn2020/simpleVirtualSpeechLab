import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
import json
from datetime import datetime
import random
from controllers.base import ControlSystem
from controllers.implementations import Controller, AbsoluteSensorProcessor, RelativeSensorProcessor
from utils.analysis import AnalysisMixin
from visualization.plotting import PlotMixin
from pitch_pert_calibration.pitchpert_dataprep import data_prep, truncate_data
from utils.signal_synth import RampedStep1D
from utils.get_configs import get_paths, get_params
from controllers.simpleDIVAtest import Controller as DIVAController
from controllers.simpleDIVAtest import get_sensor_processor
from visualization.readouts import get_params_for_implementation, readout_optimized_params
from visualization.readouts import calibration_info_pack
from utils.processing import make_jsonable_dict

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
        
    def objective_function(self, params):
        """
        Objective function for parameter optimization.
        
        Args:
            params: Flattened array of parameters to optimize
        
        Returns:
            MSE between simulation and target response
        """
        param_config, bounds, x0, current_params = calibration_info_pack(self.params_obj)
        print('current_params', current_params)
        if self.params_obj.system_type == 'DIVA':
            print('Diva sensor processor')

            system = DIVAController(self.sensor_processor, self.T_sim, self.params_obj.dt, self.pert_signal.signal, self.pert_signal.start_ramp_up, self.target_response, current_params)
            system.simulate(self.params_obj.kearney_name)

        else:
            # Template system - use existing logic
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
                sensor_processor=self.sensor_processor, 
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
        # plt.plot(timeseries_truncated, system_response_truncated)
        # plt.plot(timeseries_truncated, self.target_response)
        # plt.show()
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
        param_config, bounds, x0 = calibration_info_pack(self.params_obj)

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
        
        # Apply optimized parameters to the model
        self.apply_optimized_params(result.x)
        
        print('Optimization completed:')
        print(f'Final MSE: {result.fun}')
        print(f'Number of iterations: {result.nit}')
        print(f'Optimization success: {result.success}')
        print(f'Message: {result.message}')
        print(f'Number of MSE values recorded: {len(self.mse_history)}')
        
        plt.plot(self.mse_history)
        plt.show()
        return self.params_obj, self.mse_history
    
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
        import os
        import json
        from datetime import datetime
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend for SSH
        
        # Setup output directory
        if output_dir is None:
            from utils.get_configs import get_paths
            path_obj = get_paths()
            output_dir = path_obj.fig_save_path
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(output_dir, f"particle_swarm_run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Setup logging
        log_file = os.path.join(run_dir, "optimization_log.txt")
        progress_file = os.path.join(run_dir, "progress_history.json")
        
        def log_message(message, print_to_console=True):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] {message}"
            with open(log_file, 'a') as f:
                f.write(log_entry + '\n')
            if print_to_console:
                print(log_entry)
    
        # Initialize tracking variables 
        param_config, bounds, x0, current_params = calibration_info_pack(self.params_obj)
        print('current_params', current_params)
        bounds = np.array(bounds)  # Convert to numpy array for indexing

        num_params = len(bounds)
        best_overall_rmse = np.inf
        best_overall_params = None
        
        # Progress tracking
        progress_data = {
            'runs': [],
            'overall_best_rmse': [],
            'overall_best_params': [],
            'run_summaries': []
        }
        
        log_message(f"Starting particle swarm optimization with {runs} runs", True)
        log_message(f"Parameters: particles={num_particles}, max_iters={max_iters}, convergence_tol={convergence_tol}", True)
        log_message(f"Output directory: {run_dir}", True)
        
        start_time = datetime.now()
        
        for run in range(runs):
            run_start_time = datetime.now()
            log_message(f"Starting run {run + 1}/{runs}", True)
            
            #particles = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_particles, num_params))
            particles = self.quantized_uniform(bounds[:, 0], bounds[:, 1], precision=3, size=(num_particles, num_params))
            best_rmse = np.inf
            best_params = None
            best_history = []
            no_improvement_count = 0
            
            # Run-specific progress tracking
            run_progress = {
                'run_number': run + 1,
                'iterations': [],
                'best_rmse_history': [],
                'mean_rmse_history': [],
                'std_rmse_history': [],
                'convergence_info': {
                    'converged': False,
                    'iteration': max_iters,
                    'reason': 'Maximum iterations reached'
                }
            }
            
            for it in range(max_iters):
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
                    log_message(f"Run {run + 1}, Iter {it + 1}: New best RMSE = {best_rmse:.6f}", False)
                else:
                    no_improvement_count += 1
                
                # Log progress periodically
                if (it + 1) % log_interval == 0:
                    log_message(f"Run {run + 1}, Iter {it + 1}/{max_iters}: Best RMSE = {best_rmse:.6f}, "
                              f"Mean RMSE = {np.mean(rmses):.6f}, No improvement = {no_improvement_count}", False)
                
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
                    log_message(f"Run {run + 1} converged at iteration {it + 1}: {convergence_reason}", True)
                    run_progress['convergence_info'] = {
                        'converged': True,
                        'iteration': it + 1,
                        'reason': convergence_reason
                    }
                    break
                
                # Paper's method: Replace fraction with random linear combinations of best fits
                elite_fraction = 0.1
                replacement_fraction = 0.3

                elite_size = int(num_particles * elite_fraction)
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
            
            log_message(f"Run {run + 1} completed in {run_duration:.1f}s. Final RMSE: {best_rmse:.6f}", True)
            
            # Update overall best if this run was better
            if best_rmse < best_overall_rmse:
                best_overall_rmse = best_rmse
                best_overall_params = best_params.copy()
                log_message(f"New overall best RMSE: {best_overall_rmse:.6f}", True)
            
            # Store run summary
            progress_data['runs'].append(run + 1)
            progress_data['overall_best_rmse'].append(float(best_overall_rmse))
            progress_data['overall_best_params'].append(best_overall_params.tolist() if best_overall_params is not None else None)
            progress_data['run_summaries'].append({
                'run_number': run + 1,
                'final_rmse': float(best_rmse),
                'iterations_completed': len(best_history),
                'duration_seconds': run_duration,
                'convergence_info': run_progress['convergence_info']
            })
            
            # Save run progress
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
            
            # Create run plots
            self._create_run_plots(run_dir, run, run_progress)
        
        # All runs completed
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        log_message(f"All runs completed in {total_duration:.1f}s", True)
        log_message(f"Best overall RMSE: {best_overall_rmse:.6f}", True)
        
        # Apply the best parameters to the model
        if best_overall_params is not None:
            self.apply_optimized_params(best_overall_params)
        
        # Save final results
        self._save_final_results(run_dir, best_overall_params, best_overall_rmse, progress_data)
        
        # Return the optimized parameters, mse history, and run directory
        return self.params_obj, best_overall_rmse, run_dir

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
            config = get_params_for_implementation(self.params_obj.system_type, self.params_obj.kearney_name)
        elif self.params_obj.system_type == 'Template':
            config = get_params_for_implementation(self.params_obj.system_type)
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

    def apply_optimized_params(self, optimized_params):
        """
        Apply optimized parameters to the model.
        
        Args:
            optimized_params: Array of optimized parameter values
        """
        print('START:')
        readout_optimized_params(self.params_obj, format_opt=['print'])
        if optimized_params is not None:
            if self.params_obj.system_type == 'DIVA':
                config = get_params_for_implementation(self.params_obj.system_type, self.params_obj.kearney_name)
            elif self.params_obj.system_type == 'Template':
                config = get_params_for_implementation(self.params_obj.system_type)
            
            # Make optimized_params a dictionary
            optimized_params_dict = {
                param_name: value
                for param_name, value in zip(config, optimized_params)
            }
            # Now you can use optimized_params_dict[param_name] to access values by name
            for param_name in config:  
                if hasattr(self.params_obj, param_name):
                    setattr(self.params_obj, param_name, optimized_params_dict[param_name])
        readout_optimized_params(self.params_obj, format_opt=['print'])
        print('- END')
        
    def _apply_params_to_model(self, best_params):
        # Implementation of _apply_params_to_model method
        pass

