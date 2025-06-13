import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from controllers.base import ControlSystem
from controllers.implementations import Controller, AbsoluteSensorProcessor, RelativeSensorProcessor
from utils.analysis import AnalysisMixin
from visualization.plotting import PlotMixin
from utils.pitchpert_dataprep import data_prep, truncate_data
from utils.signal_synth import RampedStep1D
from utils.get_configs import get_paths, get_params
from utils.pitchpert_calibration import get_perturbation_event_times, PitchPertCalibrator
from visualization.readouts import readout_optimized_params

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


# Load calibration data
calibration_data = np.loadtxt(data_path_interp, delimiter=',', skiprows=1)
target_response = calibration_data[:, params_obj.trial_ID+2]  # Assuming third column is first participant's target response
pitch_pert_data = calibration_data[:, 1]

pitch_pert_rampstart, pitch_pert_rampend = get_perturbation_event_times(data_path_interp)
print('pitch_pert_rampstart', pitch_pert_rampstart)
print('pitch_pert_rampend', pitch_pert_rampend)

##Generate perturbation signal

pert_signal = RampedStep1D(params_obj.duration, dt=params_obj.dt, tstart_step=pitch_pert_rampstart, t_step_peak=pitch_pert_rampend, amp_step=params_obj.pert_mag,
                                            dist_duration=params_obj.pert_duration, ramp_up_duration=None,
                                            ramp_down_duration=params_obj.ramp_down_duration,
                                        sig_label='Step pertubation')

# plt.plot(T_sim, pert_signal.signal)
# plt.show()

# #Plot pitch pert simulation v data with initial params
# system = Controller(sensor_processor=RelativeSensorProcessor(), input_A=params_obj.A_init, input_B=params_obj.B_init, input_C=params_obj.C_aud_init, ref_type=params_obj.ref_type, K_vals=[params_obj.K_aud_init, params_obj.K_som_init], L_vals=[params_obj.L_aud_init, params_obj.L_som_init], timeseries=T_sim)    

# timeseries_truncated, system_response_truncated = truncate_data(T_sim, system.x, truncate_start, truncate_end)

# aud_pert_truncated = truncate_data(T_sim, system.v_aud, truncate_start, truncate_end)[1]

# system.simulate_with_2sensors(delta_t_s_aud=params_obj.sensor_delay_aud, delta_t_s_som=params_obj.sensor_delay_som, delta_t_a=params_obj.actuator_delay)
# system.plot_data_overlay('abs2sens', target_response, pitch_pert_data, time_trunc=timeseries_truncated, resp_trunc=system_response_truncated, pitch_pert_truncated=aud_pert_truncated, fig_save_path=fig_save_path)

calibrate_opt = False

if calibrate_opt == True:
# Create a calibrator instance
    calibrator = PitchPertCalibrator(
        params_obj=params_obj,
        target_response=target_response,
        pert_signal=pert_signal,
        T_sim=T_sim,
        truncate_start=truncate_start,
        truncate_end=truncate_end
    )

    # Run the calibration
    cal_params, mse_history = calibrator.calibrate(
        max_iterations=200,
        learning_rate=0.01,
        tolerance=1e-6
    )

    print('mse_history', mse_history)

    sensor_delay_aud = int(cal_params.sensor_delay_aud)
    sensor_delay_som = int(cal_params.sensor_delay_som)
    actuator_delay = int(cal_params.actuator_delay)

    readout_optimized_params(cal_params, sensor_delay_aud, sensor_delay_som, actuator_delay)
else:
    cal_params = params_obj
    sensor_delay_aud = int(cal_params.sensor_delay_aud)
    sensor_delay_som = int(cal_params.sensor_delay_som)
    actuator_delay = int(cal_params.actuator_delay)

#Run simulation with calibrated params
system = Controller(sensor_processor=RelativeSensorProcessor(), input_A=cal_params.A_init, input_B=cal_params.B_init, input_C=cal_params.C_aud_init, ref_type=params_obj.ref_type, K_vals=[cal_params.K_aud_init, cal_params.K_som_init], L_vals=[cal_params.L_aud_init, cal_params.L_som_init], timeseries=T_sim)    
#system.simulate_with_2sensors(delta_t_s_aud=sensor_delay_aud, delta_t_s_som=sensor_delay_som, delta_t_a=actuator_delay)
system.simulate_with_1sensor(delta_t_s=sensor_delay_aud, delta_t_a=actuator_delay)
#system.plot_transient('abs2sens', start_dist=pert_signal.start_ramp_up) 
system.plot_all('abs2sens', custom_sig='dist', fig_save_path=fig_save_path)
timeseries_truncated, system_response_truncated = truncate_data(T_sim, system.x, truncate_start, truncate_end)

aud_pert_truncated = truncate_data(T_sim, system.v_aud, truncate_start, truncate_end)[1]
system.plot_data_overlay('abs2sens', target_response, pitch_pert_data, time_trunc=timeseries_truncated, resp_trunc=system_response_truncated, pitch_pert_truncated=aud_pert_truncated, fig_save_path=fig_save_path)
#system.plot_truncated(truncate_start, truncate_end)