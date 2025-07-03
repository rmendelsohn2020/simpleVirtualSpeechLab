import numpy as np
import control as ct
from visualization.plotting import PlotMixin


class RampedStep1D(PlotMixin):
	def __init__(self, duration, dt=0.01, tstart_step=None, t_step_peak=0, amp_step=1, dist_duration=1.5, ramp_up_duration=0.5, peak_duration=None, ramp_down_duration=0.5, sig_label='Ramped step signal', units='cents', **kwargs):
		#TODO: Add documentation with expected units for each parameter
		# step_signal = vsl.signals.RampedStep1D(exp_trial.duration, dt=sec_per_step, t_step_peak=exp_trial.step_onset,amp_step=exp_trial.step,
		#                                 dist_duration=exp_trial.step_duration, ramp_up_duration=exp_trial.ramp_up_duration, 
		#                                 ramp_down_duration=exp_trial.ramp_down_duration,
		#                             sig_label='Step pertubation')
		self.dt = dt
		self.sig_label = sig_label
		self.tstart_step = tstart_step
		if units == 'multiplier':
			self.amp_step = amp_step - 1
		else:	
			self.amp_step = amp_step
		self.ramp_up_duration = ramp_up_duration
		self.ramp_down_duration = ramp_down_duration




		T_sim = np.arange(0, duration, dt)
		self.T_sim = T_sim

		if tstart_step is None:
			self.t_step_peak = t_step_peak
			self.tstart_step = t_step_peak - ramp_up_duration
			print('t_step_peak (specified):', self.t_step_peak)
			print('tstart_step:', self.tstart_step)
			
		else:
			if ramp_up_duration is None:
				ramp_up_duration = t_step_peak - tstart_step
			elif t_step_peak is None:
				t_step_peak = tstart_step + ramp_up_duration
			else:
				print('tstart_step, t_step_peak, and ramp_up_duration are all specified. Using tstart_step and ramp_up_duration to calculate t_step_peak')
			self.t_step_peak = tstart_step + ramp_up_duration
			self.tstart_step = tstart_step
			print('tstart_step (specified):', self.tstart_step)
			print('t_step_peak:', self.t_step_peak)

		if peak_duration is None:
			# Calculate peak_duration by subtracting ramp_up_duration and ramp_down_duration from dist_duration
			peak_duration = dist_duration - ramp_up_duration - ramp_down_duration
			print('peak_duration:', peak_duration)

		t_step_ind = np.searchsorted(T_sim, self.t_step_peak)
		self.t_step_ind = t_step_ind

		# Create the waveform with separate ramp-up, constant amplitude, and ramp-down
		w = np.zeros(T_sim.size)

		# Calculate indices for ramp-up and peak
		start_ramp_up = np.searchsorted(T_sim, self.tstart_step)
		end_ramp_up = t_step_ind
		start_peak = t_step_ind
		end_peak = np.searchsorted(T_sim, self.t_step_peak + peak_duration)

		# Apply linear ramp-up
		if start_ramp_up < end_ramp_up:
			# Use the sign of amp_step to determine ramp direction
			ramp_up = np.linspace(0, self.amp_step, end_ramp_up - start_ramp_up)
			w[start_ramp_up:end_ramp_up] = ramp_up

		# Set peak amplitude and maintain it for the peak duration
		w[start_peak:end_peak] = self.amp_step

		# Calculate indices for ramp-down
		start_ramp_down = end_peak
		end_ramp_down = np.searchsorted(T_sim, self.t_step_peak + peak_duration + ramp_down_duration)

		# Apply linear ramp-down
		if start_ramp_down < end_ramp_down:
			# Use the sign of amp_step to determine ramp direction
			ramp_down = np.linspace(self.amp_step, 0, end_ramp_down - start_ramp_down)
			w[start_ramp_down:end_ramp_down] = ramp_down

		if units == 'multiplier':
			w=w+1
		self.signal = w
		self.timeseries = T_sim
		self.start_ramp_up = start_ramp_up