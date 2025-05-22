import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from controllers.base import ControlSystem
from controllers.implementations import AbsEstController, RelEstController
from utils.analysis import AnalysisMixin
from visualization.plotting import PlotMixin
from utils.pitchpert_dataprep import data_prep, truncate_data
from utils.signal_synth import RampedStep1D

#Save Path
save_path = '/Users/rachelmendelsohn/Desktop/Salk/ArticulatoryFeedback/NewVSLCodebase/Interp_Data'
#Calibration Data Path
data_dir='/Users/rachelmendelsohn/Desktop/Salk/ArticulatoryFeedback/Calibration_Data/'
filename = 'secs_Smith-et-al-2020-auditory-perturbation-baseline-normalized.csv'
data_path = os.path.join(data_dir, filename)

###Define Perturbation Experiment Parameters
duration = 10.8 # Duration of each trial in seconds
sec_per_step = 0.01
T_sim = np.arange(0,duration, sec_per_step)


pert_mag = -100 # Purturbation in cents
pert_onset = 2.03 # Perturbation onset in seconds (Laryngeal 2.05 w/auditory masking, 2.06 w/o auditory masking)
pert_duration = 1.26 # Duration of perturbation in seconds
ramp_up_duration=0.178 # Duration of the ramp up in seconds
ramp_down_duration=0.250 # Duration of the ramp down in seconds

###Load Data

#Truncation to match Smith et al. 2020 data and plots
truncate = True
truncate_start = pert_onset - 0.5
truncate_end = pert_onset + 1.0

#Interpolate the calibration data
timeseries = truncate_data(T_sim, None, truncate_start, truncate_end)[0]
data_path_interp = data_prep(data_path, timeseries, save_path, convert_opt='multiplier2cents')


###Generate perturbation signal
#pert_signal = RampedStep1D(duration, sec_per_step, pert_onset, pert_mag, pert_duration, ramp_up_duration, ramp_down_duration)
pert_signal = RampedStep1D(duration, dt=sec_per_step, tstart_step=pert_onset, t_step_peak=None, amp_step=pert_mag,
                                            dist_duration=pert_duration, ramp_up_duration=ramp_up_duration, 
                                            ramp_down_duration=ramp_down_duration,
                                        sig_label='Step pertubation')
pert_signal.plot_signal(pert_signal.signal, 'Perturbation Signal')

###Run Simulation
# Simulation parameters
A = np.array([0.5])
B = np.array([1])
C = np.array([1])
ref_item = 'sin' #'sin' or 'null'
actuator_delays = 0
sensor_delays = 3

system = AbsEstController(A, B, C, ref_item, dist_custom=pert_signal.signal, dist_type='Auditory')    
system.simulate_with_2sensors()
system.plot_transient('abs2sens', start_dist=pert_signal.start_ramp_up) 
system.plot_all('abs2sens')

  