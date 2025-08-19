import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os 
from datetime import datetime

from controllers.base import ControlSystem
from controllers.implementations import Controller, AbsoluteSensorProcessor, RelativeSensorProcessor
from utils.analysis import AnalysisMixin
from visualization.plotting import PlotMixin
from pitch_pert_calibration.pitchpert_dataprep import data_prep, truncate_data
from utils.signal_synth import RampedStep1D
from utils.get_configs import get_paths, get_params
from pitch_pert_calibration.pitchpert_calibration import get_perturbation_event_times, PitchPertCalibrator
from visualization.readouts import readout_optimized_params, get_current_params, calibration_info_pack, get_params_for_implementation
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

params_obj = get_params()
calibrate_opt = params_obj.cal_set_dict['fit_method']
print('calibrate_opt', calibrate_opt)

if params_obj.system_type == 'DIVA':
    system_choice = 'DIVA'
    units = 'multiplier'
    sensor_processor = get_sensor_processor(params_obj.kearney_name)
elif params_obj.system_type == 'Template':
    units = 'cents'
    if params_obj.implementation == 'Relative':
        system_choice = 'Relative'
        sensor_processor = RelativeSensorProcessor()
    elif params_obj.implementation == 'Absolute':
        system_choice = 'Absolute'
        sensor_processor = AbsoluteSensorProcessor()
    else:
        print('No implementation specified')
        sensor_processor = None
    print('params_obj.arb_name', params_obj.arb_name)
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
# save_path = fig_save_path + '/pert_signal.png'
# plt.savefig(save_path)
# print(f"Figure saved to {save_path}")

# #Plot pitch pert simulation v data with initial params
# system = Controller(sensor_processor=RelativeSensorProcessor(), input_A=params_obj.A_init, input_B=params_obj.B_init, input_C=params_obj.C_aud_init, ref_type=params_obj.ref_type, K_vals=[params_obj.K_aud_init, params_obj.K_som_init], L_vals=[params_obj.L_aud_init, params_obj.L_som_init], timeseries=T_sim)    
def run_calibration(calibrate_opt, params_obj, target_response, pert_signal, T_sim, truncate_start, truncate_end, sensor_processor):
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
        cal_params, mse_history, run_dir = calibrator.calibrate(
            max_iterations=params_obj.cal_set_dict['max_iterations'],
            learning_rate=params_obj.cal_set_dict['learning_rate'],
            tolerance=params_obj.cal_set_dict['tolerance']
        )

        print('mse_history', mse_history)

        
        
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
            num_particles=params_obj.cal_set_dict['particle_size'],
            max_iters=params_obj.cal_set_dict['iterations'],
            convergence_tol=params_obj.cal_set_dict['tolerance'],
            runs=params_obj.cal_set_dict['runs'],
            log_interval=20,  # Log every 20 iterations
            save_interval=100,  # Save intermediate results every 100 iterations
            output_dir=None  # Uses default output directory
        )

        print('mse_history', mse_history)
        print(f'Results saved to: {run_dir}')

    elif calibrate_opt == 'PySwarms':
        calibrator = PitchPertCalibrator(
            params_obj=params_obj,
            target_response=target_response,
            pert_signal=pert_signal,
            T_sim=T_sim,
            truncate_start=truncate_start,
            truncate_end=truncate_end,
            sensor_processor=sensor_processor
        )
        cal_params, mse_history, run_dir = calibrator.pyswarms_calibrate(
            num_particles=params_obj.cal_set_dict['particle_size'],
            max_iters=params_obj.cal_set_dict['iterations'],
            convergence_tol=params_obj.cal_set_dict['tolerance'],
            runs=params_obj.cal_set_dict['runs'],
            log_interval=1,  
            save_interval=100,  
            output_dir=None,
            parallel_opt=True  # Uses default output directory
        )[0:3]

        print('mse_history', mse_history)
        print(f'Results saved to: {run_dir}')


        
    elif calibrate_opt == 'PySwarms Two Layer':
        calibrator = PitchPertCalibrator(
            params_obj=params_obj,
            target_response=target_response,
            pert_signal=pert_signal,
            T_sim=T_sim,
            truncate_start=truncate_start,
            truncate_end=truncate_end,
            sensor_processor=sensor_processor
        )
        cal_params, mse_history, run_dir = calibrator.pyswarms_twolayer_calibrate(
            upper_particles=10,    # Start small
            upper_iters=10,        # Few iterations initially
            upper_runs=1,          # Few runs initially
            lower_particles=1000,   # More particles for gains
            lower_iters=1,        # Few iterations for gains
            lower_runs=1           # Single run for gains
        )[0:3]
        
    else:
        cal_params = params_obj
        run_dir = fig_save_path + '/Sim_Run_Active' #_'+ datetime.now().strftime("%Y%m%d_%H%M%S")
        # Ensure the output directory exists when not using a calibrator that creates it
        os.makedirs(run_dir, exist_ok=True)
        
        calibrator = PitchPertCalibrator(
            params_obj=params_obj,
            target_response=target_response,
            pert_signal=pert_signal,
            T_sim=T_sim,
            truncate_start=truncate_start,
            truncate_end=truncate_end,
            sensor_processor=sensor_processor
        )
        mse_history = calibrator.eval_only(params_obj)

    if params_obj.system_type == 'DIVA':
        sensor_delay_aud = int(cal_params.tau_A)
        sensor_delay_som = int(cal_params.tau_S)
        actuator_delay = None
    else:
        sensor_delay_aud = int(cal_params.sensor_delay_aud)
        sensor_delay_som = int(cal_params.sensor_delay_som)
        actuator_delay = int(cal_params.actuator_delay)

            
    readout_optimized_params(cal_params, sensor_delay_aud, sensor_delay_som, actuator_delay, output_dir=run_dir)        

    return cal_params, mse_history, run_dir, sensor_delay_aud, sensor_delay_som, actuator_delay, pitch_pert_data

def run_simulation(cal_params, pert_signal, T_sim, truncate_start, truncate_end, sensor_processor, system_choice, sensor_delay_aud=None, sensor_delay_som=None, actuator_delay=None, run_dir=None, pitch_pert_data=None):
    params_obj = cal_params

    if run_dir is None:
        run_dir = fig_save_path # + '/Sim_Run_'+ datetime.now().strftime("%Y%m%d_%H%M%S")

    if sensor_delay_aud is None:
        try:
            sensor_delay_aud = int(cal_params.tau_A)
        except AttributeError:
            try:
                sensor_delay_aud = int(cal_params.sensor_delay_aud)
            except AttributeError:
                sensor_delay_aud = 0
    
    if sensor_delay_som is None:
        try:
            sensor_delay_som = int(cal_params.tau_S)
        except AttributeError:
            try:
                sensor_delay_som = int(cal_params.sensor_delay_som)
            except AttributeError:
                sensor_delay_som = 0
    
    if actuator_delay is None:
        try:
            actuator_delay = int(cal_params.actuator_delay)
        except AttributeError:
            actuator_delay = 0

    if system_choice == 'Relative':
        #Run simulation with calibrated params (Specify Gains)
            # Convert ParamsObject to dictionary for Controller
        
        param_config = get_params_for_implementation(cal_params.system_type, arb_name=cal_params.arb_name, null_values=params_obj.cal_set_dict['null_values'])
        print('run_simulation: null_values', params_obj.cal_set_dict['null_values'])
        params_dict = get_current_params(cal_params, param_config, cal_only=False, null_values=params_obj.cal_set_dict['null_values'])
        
        system = Controller(
                    sensor_processor=RelativeSensorProcessor(), 
                    params=params_dict,
                    ref_type=params_obj.ref_type, 
                    dist_custom=pert_signal.signal,
                    dist_type=['Auditory'],
                    timeseries=T_sim
                )
        #system = Controller(sensor_processor=RelativeSensorProcessor(), input_A=cal_params.A, input_B=cal_params.B, input_C=cal_params.C_aud, ref_type=params_obj.ref_type, dist_custom=pert_signal.signal, dist_type=['Auditory'], K_vals=[cal_params.K_aud, cal_params.K_som], L_vals=[cal_params.L_aud, cal_params.L_som], Kf_vals=[cal_params.Kf_aud, cal_params.Kf_som], timeseries=T_sim)    
        #Run simulation with calibrated params (Calculate Gains)
        #system = Controller(sensor_processor=RelativeSensorProcessor(), input_A=cal_params.A_init, input_B=cal_params.B_init, input_C=cal_params.C_aud_init, ref_type=params_obj.ref_type, dist_custom=pert_signal.signal, dist_type=['Auditory'], timeseries=T_sim)    
        system.simulate_with_2sensors(delta_t_s_aud=sensor_delay_aud, delta_t_s_som=sensor_delay_som, delta_t_a=actuator_delay)
        #system.simulate_with_1sensor(delta_t_s=sensor_delay_aud, delta_t_a=actuator_delay)
        
    elif system_choice == 'Absolute':
        # Convert ParamsObject to dictionary for Controller
        param_config = get_params_for_implementation(cal_params.system_type, arb_name=cal_params.arb_name, null_values=params_obj.cal_set_dict['null_values'])
        print('run_simulation: null_values', params_obj.cal_set_dict['null_values'])
        params_dict = get_current_params(cal_params, param_config, cal_only=False, null_values=params_obj.cal_set_dict['null_values'])
        
        system = Controller(
                    sensor_processor=AbsoluteSensorProcessor(), 
                    params=params_dict,
                    ref_type=params_obj.ref_type, 
                    dist_custom=pert_signal.signal,
                    dist_type=['Auditory'],
                    timeseries=T_sim
                )    
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
        current_params = calibration_info_pack(cal_params, cal_only=True)[3]
        system = DIVAController(sensor_processor, T_sim, params_obj.dt, pert_signal.signal, pert_signal.start_ramp_up, target_response, current_params)
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
    if system_choice == 'DIVA':
        aud_pert_truncated = truncate_data(T_sim, system.pert_P, truncate_start, truncate_end)[1]
    else:
        aud_pert_truncated = truncate_data(T_sim, system.v_aud, truncate_start, truncate_end)[1]

    return system, timeseries_truncated, system_response_truncated, aud_pert_truncated

# Only run plotting if this script is run directly (not imported)
if __name__ == "__main__":
    #Run Calibration
    cal_params, mse_history, run_dir, sensor_delay_aud, sensor_delay_som, actuator_delay, pitch_pert_data = run_calibration(calibrate_opt, params_obj, target_response, pert_signal, T_sim, truncate_start, truncate_end, sensor_processor)
    #Run Simulation
    system, timeseries_truncated, system_response_truncated, aud_pert_truncated = run_simulation(cal_params, pert_signal, T_sim, truncate_start, truncate_end, sensor_processor, system_choice, sensor_delay_aud, sensor_delay_som, actuator_delay, run_dir, pitch_pert_data)
    # Use run_dir if available (from particle swarm), otherwise use default fig_save_path
    plot_output_dir = run_dir if calibrate_opt == 'Particle Swarm' and 'run_dir' in locals() else fig_save_path
    #system.plot_all_subplots('abs2sens', custom_sig='dist', fig_save_path=fig_save_path)
    system.plot_data_overlay('abs2sens', target_response, pitch_pert_data, time_trunc=timeseries_truncated, resp_trunc=system_response_truncated, pitch_pert_truncated=aud_pert_truncated, output_dir=run_dir)
#system.plot_truncated(truncate_start, truncate_end)