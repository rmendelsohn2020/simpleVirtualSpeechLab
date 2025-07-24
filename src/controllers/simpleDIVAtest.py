import numpy as np
import scipy.integrate as integrate
from pitch_pert_calibration.pitchpert_dataprep import calculate_distortion_in_hz
from visualization.plotting import PlotMixin
from abc import ABC, abstractmethod
from utils.analysis import AnalysisMixin

#Base
class Control_System_simpleDIVA(PlotMixin):

    def __init__(self, timeseries, dt, perturbation, pert_onset, target_response, params):
        """
        Simple DIVA base
        """
        #Input parameters
        for param_name, value in params.items():
            setattr(self, param_name, value)

        self.pert_onset = pert_onset
        #self.f_Target = self.calculate_f_target(target_response, self.pert_onset)
        #self.pert_P = self.calculate_pert_P(perturbation)
        self.pert_P = perturbation
        
        self.f_Target = 1
        ###Simulation parameters
        if timeseries is not None:
            self.timeseries = timeseries
            self.time_length = len(self.timeseries)
            self.dt = dt
        else:    
            self.dt = 0.01
            self.T = 4 #seconds
            self.timeseries=np.arange(0,self.T,self.dt)  
            self.time_length = len(self.timeseries)

        ###System parameters
        self.f_A = np.zeros(self.time_length)
        self.f_S = np.zeros(self.time_length)
        self.f_Ci = np.zeros(self.time_length)
        self.f = np.zeros(self.time_length)  # Add missing f array
        
        self.f[0] = self.f_Target

#Sensor processors
def get_sensor_processor(kearney_name):
        EQ5_names = ['D1', 'D2']
        EQ6_names = ['D5', 'D6', 'D7', 'D8', 'D9', 'D10']
        EQ7_names = ['D11', 'D12', 'D13', 'D14', 'D15']
        if kearney_name in EQ5_names:
            return Process_EQ5()
        elif kearney_name in EQ6_names:
            return Process_EQ6()
        elif kearney_name in EQ7_names:
            return Process_EQ7()
        else:
            print(f'No equation found for {kearney_name}')
            return None

class DivaSensorProcessor(ABC):
    @abstractmethod
    def process_sensor_channel(self, controller, kearney_name, t):
        pass   

class Process_EQ5(DivaSensorProcessor):
    def process_sensor_channel(self, controller, kearney_name, t):
        controller.arch_str = kearney_name
  
        if kearney_name == 'D1':
            controller.alpha_S = 0
        elif kearney_name != 'D2':
            print(f'Incorrect equation for {kearney_name}')
            return

        controller.f_A[t] = controller.f[t-int(controller.tau_A)]*controller.pert_P[t-int(controller.tau_A)]
        controller.f_S[t] = controller.f[t-int(controller.tau_S)]
        if t > controller.pert_onset:
            controller.f_Ci[t] = controller.alpha_A*(controller.f_Target-controller.f_A[t]) + controller.alpha_S*(controller.f_Target-controller.f_S[t]) #EQ5
        else:
            controller.f_Ci[t] = 0 #TODO: Check this

class Process_EQ6(DivaSensorProcessor):
    def process_sensor_channel(self, controller, kearney_name, t):
        print(f'Equation 6 not yet implemented')
        return

class Process_EQ7(DivaSensorProcessor):
    def process_sensor_channel(self, controller, kearney_name, t):
        print(f'Equation 7 not yet implemented')
        return


#Controller
class Controller(Control_System_simpleDIVA, PlotMixin, AnalysisMixin):
    def __init__(self, sensor_processor: DivaSensorProcessor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sensor_processor = sensor_processor

    def process_global(self, t, channels=None):
         # Calculate control integral using cumulative sum instead of quad
        control_integral = np.sum(self.f_Ci[:t+1]) * self.dt
        self.f[t+1] = self.f_Target + control_integral

    def simulate(self, kearney_name):
        """
        Simple DIVA implementation
        """
        # List of possible tau attribute names
        tau_names = ['tau_A', 'tau_S', 'tau_As', 'tau_Ss', 'tau_Av', 'tau_Sv']

        # Get the values if they exist, otherwise None
        possible_taus = [getattr(self, name, None) for name in tau_names]

        # Filter out None values
        taus_present = [tau for tau in possible_taus if tau is not None]

        if taus_present:
            int_tau = int(max(taus_present))
        else:
            int_tau = 0  # or some other sensible default

        for t in range(0, self.time_length-1-int_tau):
            self.sensor_processor.process_sensor_channel(self, kearney_name, t)
            self.process_global(t)
        
        print('****PROCESSING DONE****')


        self.notation_conversion()


    def calculate_f_target(self, target_response, pert_onset):
        """
        Calculate the target response
        """
        baseline_range = target_response[0:pert_onset]
        f_Target = np.sum(baseline_range)/len(baseline_range)
        return f_Target
    
    # def calculate_pert_P(self, perturbation):
    #     #pert_hz = calculate_distortion_in_hz(self.f_Target, perturbation)
    #     pert_P = perturbation
    #     return pert_P

    def notation_conversion(self):
        self.y_aud = self.f_A
        self.y_som = self.f_S
        self.u = self.f_Ci
        self.x = self.f
        self.v_aud = self.pert_P
        self.ref_type = 'null'
        self.arch_str = 'D1'
        self.arch_title = 'Simple DIVA'
        
        return self.y_aud, self.y_som, self.u, self.x, self.v_aud


        

        