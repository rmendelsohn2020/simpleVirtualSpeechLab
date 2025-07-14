import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

from controllers.base import ControlSystem
from controllers.implementations import Controller, AbsoluteSensorProcessor, RelativeSensorProcessor
from utils.analysis import AnalysisMixin
from visualization.plotting import PlotMixin
from utils.pitchpert_dataprep import data_prep, truncate_data
from utils.signal_synth import RampedStep1D
from utils.get_configs import get_paths, get_params
from utils.pitchpert_calibration import get_perturbation_event_times, PitchPertCalibrator
from visualization.readouts import readout_optimized_params
from controllers.simpleDIVAtest import Controller as DIVAController
from controllers.simpleDIVAtest import get_sensor_processor
# Get experiment parameters
path_obj = get_paths()
#Save Paths
data_save_path = path_obj.data_save_path
fig_save_path = path_obj.fig_save_path
#Calibration Data Path
data_path = path_obj.data_path
print('data_path', data_path)

calibrate_opt = 'Particle Swarm'

params_obj = get_params()
if params_obj.system_type == 'DIVA':
    system_choice = 'DIVA'
    units = 'multiplier'
    sensor_processor = get_sensor_processor(params_obj.kearney_name)
elif params_obj.system_type == 'Template':
    system_choice = 'Relative'
    units = 'cents'
    sensor_processor = RelativeSensorProcessor()
else:
    print('No system type specified')
    sensor_processor = None

###Define Perturbation Experiment Parameters
T_sim = np.arange(0,params_obj.duration, params_obj.dt)
###Load Data

#Truncation to match Smith et al. 2020 data and plots
truncate = True
truncate_start = params_obj.pert_onset - 0.5
truncate_end = params_obj.pert_onset + 1.0

#Interpolate the calibration data
timeseries = truncate_data(T_sim, None, truncate_start, truncate_end)[0]

if units == 'multiplier':
    convert_opt = 'multiplier'
elif units == 'cents':
    convert_opt='multiplier2cents'
data_path_interp = data_prep(data_path, timeseries, data_save_path, convert_opt=convert_opt, pert_onset=params_obj.pert_onset, showplt=False)
# Load calibration data
calibration_data = np.loadtxt(data_path_interp, delimiter=',', skiprows=1)
target_response = calibration_data[:, params_obj.trial_ID+2]  # Assuming third column is first participant's target response
pitch_pert_data = calibration_data[:, 1]

# plt.plot(timeseries, target_response)
# plt.show()

pitch_pert_rampstart, pitch_pert_rampend, max_val = get_perturbation_event_times(data_path_interp, units=units)
print('pitch_pert_rampstart', pitch_pert_rampstart)
print('pitch_pert_rampend', pitch_pert_rampend)

##Generate perturbation signal

pert_signal = RampedStep1D(params_obj.duration, dt=params_obj.dt, tstart_step=pitch_pert_rampstart, t_step_peak=pitch_pert_rampend, amp_step=max_val,
                                            dist_duration=params_obj.pert_duration, ramp_up_duration=None,
                                            ramp_down_duration=params_obj.ramp_down_duration,
                                        sig_label='Step pertubation', units=units)

# plt.plot(T_sim, pert_signal.signal)
# plt.show()


if calibrate_opt == 'Standard':
# Create a calibrator instance
    calibrator = PitchPertCalibrator(
        params_obj=params_obj,
        target_response=target_response,
        pert_signal=pert_signal,
        T_sim=T_sim,
        truncate_start=truncate_start,
        truncate_end=truncate_end,
        sensor_processor=sensor_processor
    )

    # Run the calibration
    cal_params, mse_history = calibrator.calibrate(
        max_iterations=400,
        learning_rate=0.01,
        tolerance=1e-6
    )

    print('mse_history', mse_history)
    if params_obj.system_type == 'DIVA':
        sensor_delay_aud = int(cal_params.tau_A)
        sensor_delay_som = int(cal_params.tau_S)
    else:
        sensor_delay_aud = int(cal_params.sensor_delay_aud)
        sensor_delay_som = int(cal_params.sensor_delay_som)
        actuator_delay = int(cal_params.actuator_delay)

    readout_optimized_params(cal_params, sensor_delay_aud, sensor_delay_som, actuator_delay)
elif calibrate_opt == 'Particle Swarm':
    calibrator = PitchPertCalibrator(
        params_obj=params_obj,
        target_response=target_response,
        pert_signal=pert_signal,
        T_sim=T_sim,
        truncate_start=truncate_start,
        truncate_end=truncate_end,
        sensor_processor=sensor_processor
    )

    cal_params, mse_history, run_dir = calibrator.particle_swarm_calibrate(
        num_particles=100,
        max_iters=10,
        convergence_tol=0.01,
        runs=1,
        log_interval=20,  # Log every 20 iterations
        save_interval=100,  # Save intermediate results every 100 iterations
        output_dir=None  # Uses default output directory
    )

    print('mse_history', mse_history)
    print(f'Results saved to: {run_dir}')

    if params_obj.system_type == 'DIVA':
        sensor_delay_aud = int(cal_params.tau_A)
        sensor_delay_som = int(cal_params.tau_S)
        actuator_delay = None
    else:
        sensor_delay_aud = int(cal_params.sensor_delay_aud)
        sensor_delay_som = int(cal_params.sensor_delay_som)
        actuator_delay = int(cal_params.actuator_delay)

    # Save optimized parameters to the timestamped folder
    readout_optimized_params(cal_params, sensor_delay_aud, sensor_delay_som, actuator_delay, output_dir=run_dir)
elif calibrate_opt == 'DIVA':
    cal_params = params_obj
    print('DIVA calibration not yet implemented')
else:
    cal_params = params_obj

    if params_obj.system_type == 'DIVA':
        sensor_delay_aud = int(cal_params.tau_A)
        sensor_delay_som = int(cal_params.tau_S)
        actuator_delay = None
    else:
        sensor_delay_aud = int(cal_params.sensor_delay_aud)
        sensor_delay_som = int(cal_params.sensor_delay_som)
        actuator_delay = int(cal_params.actuator_delay)



if system_choice == 'Relative':
    #Run simulation with calibrated params (Specify Gains)
    system = Controller(sensor_processor=RelativeSensorProcessor(), input_A=cal_params.A_init, input_B=cal_params.B_init, input_C=cal_params.C_aud_init, ref_type=params_obj.ref_type, dist_custom=pert_signal.signal, dist_type=['Auditory'], K_vals=[cal_params.K_aud_init, cal_params.K_som_init], L_vals=[cal_params.L_aud_init, cal_params.L_som_init], Kf_vals=[cal_params.Kf_aud_init, cal_params.Kf_som_init], timeseries=T_sim)    
    #Run simulation with calibrated params (Calculate Gains)
    #system = Controller(sensor_processor=RelativeSensorProcessor(), input_A=cal_params.A_init, input_B=cal_params.B_init, input_C=cal_params.C_aud_init, ref_type=params_obj.ref_type, dist_custom=pert_signal.signal, dist_type=['Auditory'], timeseries=T_sim)    
    system.simulate_with_2sensors(delta_t_s_aud=sensor_delay_aud, delta_t_s_som=sensor_delay_som, delta_t_a=actuator_delay)
    #system.simulate_with_1sensor(delta_t_s=sensor_delay_aud, delta_t_a=actuator_delay)
elif system_choice == 'Absolute':
    system = Controller(sensor_processor=AbsoluteSensorProcessor(), input_A=cal_params.A_init, input_B=cal_params.B_init, input_C=cal_params.C_aud_init, ref_type=params_obj.ref_type, dist_custom=pert_signal.signal, dist_type=['Auditory'], K_vals=[cal_params.K_aud_init, cal_params.K_som_init], L_vals=[cal_params.L_aud_init, cal_params.L_som_init], Kf_vals=[cal_params.Kf_aud_init, cal_params.Kf_som_init], timeseries=T_sim)    
    system.simulate_with_2sensors(delta_t_s_aud=sensor_delay_aud, delta_t_s_som=sensor_delay_som, delta_t_a=actuator_delay)
elif system_choice == 'DIVA':
    # alpha_A = 2.0
    # alpha_S = 3.0
    # alpha_Av = None
    # alpha_Sv = None
    # tau_A = 0.128
    # tau_A_int = round(cal_params.tau_A_init/params_obj.dt)
    # tau_S_int = round(cal_params.tau_S_init/params_obj.dt)
    # tau_As_int = round(cal_params.tau_As_init/params_obj.dt)
    # tau_Ss_int = round(cal_params.tau_Ss_init/params_obj.dt)
    # tau_S = 0

    # Get the appropriate sensor processor for the DIVA controller
    sensor_processor = get_sensor_processor(cal_params.kearney_name)
    system = DIVAController(sensor_processor, T_sim, params_obj.dt, pert_signal.signal, pert_signal.start_ramp_up, target_response, cal_params.alpha_A_init, cal_params.alpha_S_init, cal_params.alpha_Av_init, cal_params.alpha_Sv_init, cal_params.tau_A, cal_params.tau_S, cal_params.tau_As, cal_params.tau_Ss)
    system.simulate(cal_params.kearney_name)

    print('system.timeseries', system.timeseries.shape)
    print('system.f', system.f.shape)
    print('system.f_A', system.f_A.shape)
    print('system.f_S', system.f_S.shape)
    print('system.f_Ci', system.f_Ci.shape)
    print('system.pert_P', system.pert_P.shape)
    # print('system.pert_P', system.pert_P)
    print('f_Target', system.f_Target)

    # plt.plot(system.timeseries, system.pert_P)
    # plt.plot(system.timeseries, system.f)
    # plt.show()

#system.plot_transient('abs2sens', start_dist=pert_signal.start_ramp_up) 
#system.plot_all('abs2sens', custom_sig='dist', fig_save_path=fig_save_path)

timeseries_truncated, system_response_truncated = truncate_data(T_sim, system.x, truncate_start, truncate_end)
aud_pert_truncated = truncate_data(T_sim, system.v_aud, truncate_start, truncate_end)[1]

# Only run plotting if this script is run directly (not imported)
if __name__ == "__main__":
    # Use run_dir if available (from particle swarm), otherwise use default fig_save_path
    plot_output_dir = run_dir if calibrate_opt == 'Particle Swarm' and 'run_dir' in locals() else fig_save_path
    system.plot_data_overlay('abs2sens', target_response, pitch_pert_data, time_trunc=timeseries_truncated, resp_trunc=system_response_truncated, pitch_pert_truncated=aud_pert_truncated, output_dir=plot_output_dir)
#system.plot_truncated(truncate_start, truncate_end)