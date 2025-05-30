import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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



def objective_function(params, params_obj, target_response, pert_signal, T_sim, truncate_start, truncate_end):
    """
    Objective function for parameter optimization.
    
    Args:
        params: Flattened array of parameters to optimize
        params_obj: Object containing system parameters
        target_response: Target response data to match
        pert_signal: Perturbation signal
        T_sim: Time series for simulation
        truncate_start: Start time for truncation
        truncate_end: End time for truncation
    
    Returns:
        MSE between simulation and target response
    """
    # Unpack parameters
    A = np.array([params[0]])
    B = np.array([params[1]])
    C = np.array([params[2]])
    K_aud = params[3]
    L_aud = params[4]
    K_som = params[5]
    L_som = params[6]
    sensor_delay_aud = params[7]
    sensor_delay_som = params[8]
    actuator_delay = params[9]
    
    # Create system with current parameters
    system = AbsEstController(
        A, B, C,
        params_obj.ref_type,
        dist_custom=pert_signal.signal,
        dist_type=['Auditory'],
        timeseries=T_sim,
        K_vals=[K_aud, K_som],
        L_vals=[L_aud, L_som]
    )
    
    # Run simulation
    system.simulate_with_2sensors(
        delta_t_s_aud=int(sensor_delay_aud/params_obj.dt),
        delta_t_s_som=int(sensor_delay_som/params_obj.dt),
        delta_t_a=int(actuator_delay/params_obj.dt)
    )
    
    # Calculate MSE between simulation and calibration data
    timeseries_truncated, system_response_truncated = truncate_data(T_sim, system.x, truncate_start, truncate_end)
    return system.mse(system_response_truncated, target_response)

def callback_function(xk, *args):
    """
    Callback function to track optimization progress.
    
    Args:
        xk: Current parameter values
        *args: Additional arguments that may be passed by different optimization methods
    """
    # Calculate MSE for current parameters
    current_mse = objective_function(xk, params_obj, target_response, pert_signal, T_sim, truncate_start, truncate_end)
    # Store the current MSE value
    callback_function.mse_history.append(current_mse)
    return False  # Return False to continue optimization

def calibrate_params(params_obj, target_response, max_iterations=100, learning_rate=0.01, tolerance=1e-6):
    """
    Calibrate system parameters to match calibration data using scipy.optimize.minimize.
    
    Args:
        params_obj: Object containing system parameters
        target_response: Target response data to match
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
        (1e-6, 5.0),      # C
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
        params_obj.A_init[0],
        params_obj.B_init[0],
        params_obj.C_init[0],
        params_obj.K_aud_init,
        params_obj.L_aud_init,
        params_obj.K_som_init,
        params_obj.L_som_init,
        params_obj.sensor_delay_aud,
        params_obj.sensor_delay_som,
        params_obj.actuator_delay
    ]
    
    # Initialize callback function's MSE history
    callback_function.mse_history = []
    
    # Run optimization
    result = minimize(
        objective_function,
        x0,
        args=(params_obj, target_response, pert_signal, T_sim, truncate_start, truncate_end),
        method='trust-constr',
        bounds=bounds,
        options={'maxiter': max_iterations},
        callback=callback_function
    )
    
    # Update parameters object with optimized values
    params_obj.A_init = np.array([result.x[0]])
    params_obj.B_init = np.array([result.x[1]])
    params_obj.C_init = np.array([result.x[2]])
    params_obj.K_aud_init = result.x[3]
    params_obj.L_aud_init = result.x[4]
    params_obj.K_som_init = result.x[5]
    params_obj.L_som_init = result.x[6]
    params_obj.sensor_delay_aud = result.x[7]
    params_obj.sensor_delay_som = result.x[8]
    params_obj.actuator_delay = result.x[9]
    
    # Get MSE history from callback
    mse_history = callback_function.mse_history
    
    print('Optimization completed:')
    print(f'Final MSE: {result.fun}')
    print(f'Number of iterations: {result.nit}')
    print(f'Optimization success: {result.success}')
    print(f'Message: {result.message}')
    print(f'Number of MSE values recorded: {len(mse_history)}')
    
    return params_obj, mse_history

# Load calibration data
calibration_data = np.loadtxt(data_path_interp, delimiter=',', skiprows=1)
target_response = calibration_data[:, params_obj.trial_ID+2]  # Assuming third column is first participant's target response

cal_params, mse_history = calibrate_params(params_obj, target_response)

print('mse_history', mse_history)

sensor_delay_aud = int(cal_params.sensor_delay_aud/params_obj.dt)
sensor_delay_som = int(cal_params.sensor_delay_som/params_obj.dt)
actuator_delay = int(cal_params.actuator_delay/params_obj.dt)

print('Optimized parameters:')
print(f'A: {cal_params.A_init}')
print(f'B: {cal_params.B_init}')
print(f'C: {cal_params.C_init}')
print(f'K_aud: {cal_params.K_aud_init}')
print(f'L_aud: {cal_params.L_aud_init}')
print(f'K_som: {cal_params.K_som_init}')
print(f'L_som: {cal_params.L_som_init}') 
print(f'sensor_delay_aud: {sensor_delay_aud}')
print(f'sensor_delay_som: {sensor_delay_som}')
print(f'actuator_delay: {actuator_delay}')

#Run simulation with calibrated params
system = AbsEstController(cal_params.A_init, cal_params.B_init, cal_params.C_init, params_obj.ref_type, dist_custom=pert_signal.signal, dist_type=['Auditory','Somatosensory'], timeseries=T_sim, K_vals=[cal_params.K_aud_init, cal_params.K_som_init], L_vals=[cal_params.L_aud_init, cal_params.L_som_init])    
system.simulate_with_2sensors(delta_t_s_aud=sensor_delay_aud, delta_t_s_som=sensor_delay_som, delta_t_a=actuator_delay)
#system.plot_transient('abs2sens', start_dist=pert_signal.start_ramp_up) 
system.plot_all('abs2sens', custom_sig='dist', fig_save_path=fig_save_path)
timeseries_truncated, system_response_truncated = truncate_data(T_sim, system.x, truncate_start, truncate_end)
system.plot_data_overlay('abs2sens', target_response, time_trunc=timeseries_truncated, resp_trunc=system_response_truncated, fig_save_path=fig_save_path)
#system.plot_truncated(truncate_start, truncate_end)