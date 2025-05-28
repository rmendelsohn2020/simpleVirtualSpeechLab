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

#TODO: Make config file that doesn't get pushed, add to gitignore
path_obj = get_paths()
#Save Paths
data_save_path = path_obj.data_save_path
fig_save_path = path_obj.fig_save_path
#Calibration Data Path
data_path = path_obj.data_path
print('data_path', data_path)

params_obj = get_params()

#TODO: Experiment config file
###Define Perturbation Experiment Parameters
T_sim = np.arange(0,params_obj.duration, params_obj.dt)

###Load Data

#Truncation to match Smith et al. 2020 data and plots
truncate = True
truncate_start = params_obj.pert_onset - 0.5
truncate_end = params_obj.pert_onset + 1.0

#Interpolate the calibration data
timeseries = truncate_data(T_sim, None, truncate_start, truncate_end)[0]
data_path_interp = data_prep(data_path, timeseries, data_save_path, convert_opt='multiplier2cents')


###Generate perturbation signal
#pert_signal = RampedStep1D(duration, sec_per_step, pert_onset, pert_mag, pert_duration, ramp_up_duration, ramp_down_duration)
pert_signal = RampedStep1D(params_obj.duration, dt=params_obj.dt, tstart_step=params_obj.pert_onset, t_step_peak=None, amp_step=params_obj.pert_mag,
                                            dist_duration=params_obj.pert_duration, ramp_up_duration=params_obj.ramp_up_duration, 
                                            ramp_down_duration=params_obj.ramp_down_duration,
                                        sig_label='Step pertubation')
#pert_signal.plot_signal(pert_signal.signal, 'Perturbation Signal')

###Run Simulation
# Simulation parameters
A = np.array(params_obj.A_init)
B = np.array(params_obj.B_init)
C = np.array(params_obj.C_init)

R_val=params_obj.R_init
RN_val=params_obj.RN_init

ref_item = 'null' #'sin' or 'null'
actuator_delay = int(params_obj.actuator_delay/params_obj.dt)
sensor_delay_aud = int(params_obj.sensor_delay_aud/params_obj.dt)
sensor_delay_som = int(params_obj.sensor_delay_som/params_obj.dt)

system = AbsEstController(params_obj.A_init, params_obj.B_init, params_obj.C_init, params_obj.ref_type, dist_custom=pert_signal.signal, dist_type=['Auditory','Somatosensory'], timeseries=T_sim, tune_R=params_obj.R_init, tune_RN=params_obj.RN_init)    
system.simulate_with_2sensors(delta_t_s_aud=sensor_delay_aud, delta_t_s_som=sensor_delay_som, delta_t_a=actuator_delay)
#system.plot_transient('abs2sens', start_dist=pert_signal.start_ramp_up) 
system.plot_all('abs2sens', custom_sig='dist', fig_save_path=fig_save_path)
#system.plot_truncated(truncate_start, truncate_end)
  