import numpy as np
import scipy.integrate as integrate
from utils.pitchpert_dataprep import calculate_distortion_in_hz
from visualization.plotting import PlotMixin
from abc import ABC, abstractmethod

#Base
class Control_System_simpleDIVA(PlotMixin):

    def __init__(self, timeseries, dt, perturbation, pert_onset, target_response, alpha_A, alpha_S, alpha_Av=None, alpha_Sv=None, alpha_As=None, alpha_Ss=None, tau_A=1, tau_S=1):
        """
        Simple DIVA base
        """
        #Input parameters
        self.alpha_A = alpha_A
        self.alpha_S = alpha_S
        if alpha_Av is not None:
            self.alpha_Av = alpha_Av
        if alpha_Sv is not None:
            self.alpha_Sv = alpha_Sv
        if alpha_As is not None:
            self.alpha_As = alpha_As
        if alpha_Ss is not None:
            self.alpha_Ss = alpha_Ss
        self.tau_A = tau_A
        self.tau_S = tau_S

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

        controller.f_A[t] = controller.f[t]*controller.pert_P[t]
        controller.f_S[t] = controller.f[t]
        if t > controller.pert_onset:
            controller.f_Ci[t] = controller.alpha_A*(controller.f_Target-controller.f_A[t]) + controller.alpha_S*(controller.f_Target-controller.f_S[t]) #EQ5
        else:
            controller.f_Ci[t] = 0 #TODO: Check this
        print('f_Ci', controller.f_Ci[t])

class Process_EQ6(DivaSensorProcessor):
    def process_sensor_channel(self, controller, kearney_name, t):
        print(f'Equation 6 not yet implemented')
        return

class Process_EQ7(DivaSensorProcessor):
    def process_sensor_channel(self, controller, kearney_name, t):
        print(f'Equation 7 not yet implemented')
        return


#Controller
class Controller(Control_System_simpleDIVA, PlotMixin):
    def __init__(self, sensor_processor: DivaSensorProcessor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sensor_processor = sensor_processor

    def process_global(self, t, channels=None):
         # Calculate control integral using cumulative sum instead of quad
        control_integral = np.sum(self.f_Ci[:t+1]) * self.dt
        print('control_integral', control_integral)
        self.f[t+1] = self.f_Target + control_integral

    def simulate(self, kearney_name):
        """
        Simple DIVA implementation
        """
        
        for t in range(0, self.time_length-1-(max(self.tau_A, self.tau_S))):
            print('t', t)
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
        return self.y_aud, self.y_som, self.u, self.x, self.v_aud


        

        