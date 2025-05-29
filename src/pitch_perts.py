import numpy as np
import matplotlib.pyplot as plt

from controllers.base import ControlSystem
from controllers.implementations import AbsEstController, RelEstController
from utils.analysis import AnalysisMixin
from visualization.plotting import PlotMixin
from utils.pitchpert_dataprep import data_prep, truncate_data
from utils.signal_synth import RampedStep1D
from utils.get_configs import get_paths, get_params

# Get experiment parameters
path_obj = get_paths()
#Save Paths
data_save_path = path_obj.data_save_path
fig_save_path = path_obj.fig_save_path
#Calibration Data Path
data_path = path_obj.data_path
print('data_path', data_path)

params_obj = get_params()

###Define Perturbation Experiment Parameters
T_sim = np.arange(0,params_obj.duration, params_obj.dt)

###Load Data

#Truncation to match Smith et al. 2020 data and plots
truncate = True
truncate_start = params_obj.pert_onset - 0.5
truncate_end = params_obj.pert_onset + 1.0

#Interpolate the calibration data
timeseries = truncate_data(T_sim, None, truncate_start, truncate_end)[0]
data_path_interp = data_prep(data_path, timeseries, data_save_path, convert_opt='multiplier2cents', pert_onset=params_obj.pert_onset)




###Generate perturbation signal
#pert_signal = RampedStep1D(duration, sec_per_step, pert_onset, pert_mag, pert_duration, ramp_up_duration, ramp_down_duration)
pert_signal = RampedStep1D(params_obj.duration, dt=params_obj.dt, tstart_step=params_obj.pert_onset, t_step_peak=None, amp_step=params_obj.pert_mag,
                                            dist_duration=params_obj.pert_duration, ramp_up_duration=params_obj.ramp_up_duration, 
                                            ramp_down_duration=params_obj.ramp_down_duration,
                                        sig_label='Step pertubation')
#pert_signal.plot_signal(pert_signal.signal, 'Perturbation Signal')




###Run Simulation
# Simulation parameters
# A = np.array(params_obj.A_init)
# B = np.array(params_obj.B_init)
# C = np.array(params_obj.C_init)



# ref_item = 'null' #'sin' or 'null'
# actuator_delay = int(params_obj.actuator_delay/params_obj.dt)
# sensor_delay_aud = int(params_obj.sensor_delay_aud/params_obj.dt)
# sensor_delay_som = int(params_obj.sensor_delay_som/params_obj.dt)




###Calibration
#Start with initial params
#Run simulation
#Calculate mse
#Adjust params
#Repeat



def calibrate_params(params_obj, target_response, max_iterations=100, learning_rate=0.01, tolerance=1e-4):
    """
    Calibrate system parameters to match calibration data using gradient descent.
    
    Args:
        params_obj: Object containing system parameters
        target_response: Target response data to match
        max_iterations: Maximum number of optimization iterations
        learning_rate: Step size for parameter updates
        tolerance: Convergence threshold for mse improvement
    
    Returns:
        Optimized parameters object
    """

    
    # Initialize optimization
    best_mse = float('inf')
    iteration = 0
    converged = False
    
    # Define delay parameter bounds (in seconds)
    DELAY_BOUNDS = {
        'sensor_delay_aud': (0.0, 10),  # time steps
        'sensor_delay_som': (0.0, 10),  # time steps
        'actuator_delay': (0.0, 10)     # time steps
    }
    
    while not converged and iteration < max_iterations:
        # Create system with current parameters
        system = AbsEstController(
            params_obj.A_init, 
            params_obj.B_init, 
            params_obj.C_init, 
            params_obj.ref_type,
            dist_custom=pert_signal.signal, 
            dist_type=['Auditory'],
            timeseries=T_sim,
            K_vals=[params_obj.K_aud_init, params_obj.K_som_init],
            L_vals=[params_obj.L_aud_init, params_obj.L_som_init],
        )
        
        # Run simulation
        system.simulate_with_2sensors(
            delta_t_s_aud=int(params_obj.sensor_delay_aud/params_obj.dt),
            delta_t_s_som=int(params_obj.sensor_delay_som/params_obj.dt),
            delta_t_a=int(params_obj.actuator_delay/params_obj.dt)
        )
        
        # Calculate mse between simulation and calibration data
        timeseries_truncated, system_response_truncated = truncate_data(T_sim, system.x, truncate_start, truncate_end)
        current_mse = system.mse(system_response_truncated, target_response)
        
        #TODO:Plot simulation and calibration data
        if iteration == 0:
            system.plot_data_overlay('abs2sens', target_response, time_trunc=timeseries_truncated, resp_trunc=system_response_truncated, fig_save_path=fig_save_path)
        # Check for convergence
        print('Checkpoint 0')    

        print(f"Iteration {iteration}, mse: {current_mse}")
        print(f"best_mse: {best_mse}")
        print(f"tolerance: {tolerance}")
        
        print('Checkpoint 0.5')    
        if abs(current_mse - best_mse) < tolerance:
            converged = True
            print(f"Converged after {iteration} iterations with mse: {current_mse}")
            break
        print('Checkpoint 1')    
        # Update best mse
        if current_mse < best_mse:
            best_mse = current_mse
            best_params = {
                'A': params_obj.A_init.copy(),
                'B': params_obj.B_init.copy(),
                'C': params_obj.C_init.copy(),
                'K_aud': params_obj.K_aud_init,
                'L_aud': params_obj.L_aud_init,
                'K_som': params_obj.K_som_init,
                'L_som': params_obj.L_som_init,
                'sensor_delay_aud': params_obj.sensor_delay_aud,
                'sensor_delay_som': params_obj.sensor_delay_som,
                'actuator_delay': params_obj.actuator_delay
            }
        print('Checkpoint 2')    
        # Calculate gradients (simplified finite difference approximation)
        params_to_optimize = ['A', 'B', 'C', 'K_aud', 'L_aud', 'K_som', 'L_som', 'sensor_delay_aud', 'sensor_delay_som', 'actuator_delay']
        for param in params_to_optimize:
            # Create perturbed parameters
            perturbed_params = params_obj.__dict__.copy()
            param_name = f'{param}_init' if param not in ['sensor_delay_aud', 'sensor_delay_som', 'actuator_delay'] else param
            
            # Perturb parameter
            if param in ['A', 'B', 'C']:
                perturbed_params[param_name] = perturbed_params[param_name] * (1 + learning_rate)
            elif param in ['sensor_delay_aud', 'sensor_delay_som', 'actuator_delay']:
                # For delay parameters, perturb by a small time step
                perturbed_params[param] = perturbed_params[param] + params_obj.dt
            else:
                perturbed_params[param_name] = perturbed_params[param_name] * (1 + learning_rate)
            
            # Run simulation with perturbed parameters
            perturbed_system = AbsEstController(
                perturbed_params['A_init'],
                perturbed_params['B_init'],
                perturbed_params['C_init'],
                perturbed_params['ref_type'],
                dist_custom=pert_signal.signal, 
                dist_type=['Auditory'],
                timeseries=T_sim,
                K_vals=[perturbed_params['K_aud_init'], perturbed_params['K_som_init']],
                L_vals=[perturbed_params['L_aud_init'], perturbed_params['L_som_init']]
            )
            
            perturbed_system.simulate_with_2sensors(
                delta_t_s_aud=int(perturbed_params['sensor_delay_aud']/params_obj.dt),
                delta_t_s_som=int(perturbed_params['sensor_delay_som']/params_obj.dt),
                delta_t_a=int(perturbed_params['actuator_delay']/params_obj.dt)
            )
            
            # Calculate gradient
            system_response_truncated = truncate_data(T_sim, perturbed_system.x, truncate_start, truncate_end)[1]
            perturbed_mse = perturbed_system.mse(system_response_truncated, target_response)
            
            # Check for NaN values in MSE calculations
            if np.isnan(perturbed_mse) or np.isnan(current_mse):
                print(f"Warning: NaN detected in MSE calculation for parameter {param}")
                print(f"Current MSE: {current_mse}, Perturbed MSE: {perturbed_mse}")
                continue
                
            gradient = (perturbed_mse - current_mse) / learning_rate
            
            # Check for NaN in gradient
            if np.isnan(gradient):
                print(f"Warning: NaN detected in gradient calculation for parameter {param}")
                print(f"Gradient value: {gradient}")
                continue
            
            # Update parameters using setattr with bounds
            current_value = getattr(params_obj, param_name)
            new_value = current_value - learning_rate * gradient
            
            # Check for NaN in new value
            if np.isnan(new_value):
                print(f"Warning: NaN detected in new value calculation for parameter {param}")
                print(f"Current value: {current_value}, New value: {new_value}")
                continue
            
            # Apply appropriate bounds based on parameter type
            if param in ['A']:
                new_value = np.clip(new_value, 1e-6, 1.1)
            elif param in ['B', 'C']:
                new_value = np.clip(new_value, 1e-6, 5.0)
            elif param in ['sensor_delay_aud', 'sensor_delay_som', 'actuator_delay']:
                # For delay parameters, ensure they stay within bounds and are multiples of dt
                lower_bound, upper_bound = DELAY_BOUNDS[param]
                new_value = np.clip(new_value, lower_bound, upper_bound)
                # Round to nearest dt, with NaN check
                if not np.isnan(new_value):
                    new_value = round(new_value / params_obj.dt) * params_obj.dt
                else:
                    print(f"Warning: NaN value detected before rounding for delay parameter {param}")
                    continue
            elif param in ['K_aud', 'L_aud', 'K_som', 'L_som']:
                new_value = np.clip(new_value, 1e-6, 1.0)
            else:
                new_value = max(new_value, 1e-6)

            # Final NaN check before setting the value
            if not np.isnan(new_value):
                print(f'new_value_clipped for {param}:', new_value)
                setattr(params_obj, param_name, new_value)
            else:
                print(f"Warning: NaN value detected after all processing for parameter {param}")

        iteration += 1
        if iteration % 10 == 0:
            print(f"Iteration {iteration}, mse: {current_mse}")
            print(f"Current delays: aud={params_obj.sensor_delay_aud:.3f}s, som={params_obj.sensor_delay_som:.3f}s, act={params_obj.actuator_delay:.3f}s")
    
    if not converged:
        print(f"Did not converge after {max_iterations} iterations")
        # Use best parameters found
        params_obj.A_init = best_params['A']
        params_obj.B_init = best_params['B']
        params_obj.C_init = best_params['C']
        params_obj.K_aud_init = best_params['K_aud']
        params_obj.L_aud_init = best_params['L_aud']
        params_obj.K_som_init = best_params['K_som']
        params_obj.L_som_init = best_params['L_som']
        params_obj.sensor_delay_aud = best_params['sensor_delay_aud']
        params_obj.sensor_delay_som = best_params['sensor_delay_som']
        params_obj.actuator_delay = best_params['actuator_delay']

        print('sensor_delay_aud (s)', params_obj.sensor_delay_aud)
        print('sensor_delay_som (s)', params_obj.sensor_delay_som)
        print('actuator_delay (s)', params_obj.actuator_delay)
    return params_obj

# Load calibration data
calibration_data = np.loadtxt(data_path_interp, delimiter=',', skiprows=1)
target_response = calibration_data[:, params_obj.trial_ID+2]  # Assuming third column is first participant's target response

cal_params = calibrate_params(params_obj, target_response)


sensor_delay_aud = int(cal_params.sensor_delay_aud/params_obj.dt)
sensor_delay_som = int(cal_params.sensor_delay_som/params_obj.dt)
actuator_delay = int(cal_params.actuator_delay/params_obj.dt)

#Run simulation with calibrated params
system = AbsEstController(cal_params.A_init, cal_params.B_init, cal_params.C_init, params_obj.ref_type, dist_custom=pert_signal.signal, dist_type=['Auditory','Somatosensory'], timeseries=T_sim, K_vals=[cal_params.K_aud_init, cal_params.K_som_init], L_vals=[cal_params.L_aud_init, cal_params.L_som_init])    
system.simulate_with_2sensors(delta_t_s_aud=sensor_delay_aud, delta_t_s_som=sensor_delay_som, delta_t_a=actuator_delay)
#system.plot_transient('abs2sens', start_dist=pert_signal.start_ramp_up) 
system.plot_all('abs2sens', custom_sig='dist', fig_save_path=fig_save_path)
timeseries_truncated, system_response_truncated = truncate_data(T_sim, system.x, truncate_start, truncate_end)
system.plot_data_overlay('abs2sens', target_response, time_trunc=timeseries_truncated, resp_trunc=system_response_truncated, fig_save_path=fig_save_path)
#system.plot_truncated(truncate_start, truncate_end)