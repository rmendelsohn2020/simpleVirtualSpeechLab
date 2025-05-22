import numpy as np
from controllers.base import ControlSystem
from controllers.implementations import AbsEstController, RelEstController
from utils.analysis import AnalysisMixin
from visualization.plotting import PlotMixin

class AbsEstController(ControlSystem, PlotMixin, AnalysisMixin):
    def simulate(self, delta_t_s=1, delta_t_a=1):
        if delta_t_s == 0 and delta_t_a == 0:
            self.arch_title = 'Absolute Est. System, No Delays'
        else:
            self.arch_title = 'Absolute Est. System, Delays (Sensor Delay: ' + str(delta_t_s) + ', Actuator Delay: ' + str(delta_t_a) + ')'
        for t in range(0,self.time_length-1-(delta_t_s+1+delta_t_a)):
            #Brain Implementation
            self.y_tilde[t]=self.y[t]-self.C*self.x_hat[t]-self.L_del*self.x_s[t]
            self.x_s[t+delta_t_s]=self.y_tilde[t]
            self.x_hat[t+delta_t_s+1] = (self.L1*self.x_s[t+delta_t_s])+(self.A*self.x_hat[t+delta_t_s])+(self.B*self.x_a[t+delta_t_s])+(self.B*self.Kf)*self.r[t+delta_t_s]
            self.x_a[t+delta_t_s+1+delta_t_a]=-self.K2*self.x_hat[t+delta_t_s+1]-self.K_del*self.x_a[t+delta_t_s+1]+self.Kf*self.r[t+delta_t_s+1]
            self.u[t+delta_t_s+1+delta_t_a]= self.x_a[t+delta_t_s+1+delta_t_a]

            #World
            self.x[t+1]=self.A*self.x[t]+self.B*self.u[t]+self.w[t]
            self.y[t+1]=self.C*self.x[t+1]+self.v[t+1]

    def simulate_with_2sensors(self, delta_t_s_aud=1, delta_t_s_som=1, delta_t_a=1):
        if delta_t_s_aud == 0 and delta_t_s_som == 0 and delta_t_a == 0:
            self.arch_title = 'Absolute Est. 2-Sensor System, No Delays'
        else:
            self.arch_title = 'Absolute Est. 2-Sensor System, Delays (Auditory Sensor Delay: ' + str(delta_t_s_aud) + ', Somatosensory Sensor Delay: ' + str(delta_t_s_som) + ', Actuator Delay: ' + str(delta_t_a) + ')'
        for t in range(0,self.time_length-1-(delta_t_s_aud+1+delta_t_s_som+1+delta_t_a)):
            #Brain Implementation
            ##Auditory Sensor
            self.y_aud[t] = self.y[t]+self.v_aud[t] #Placeholder
            self.y_tilde_aud[t]=(self.y[t])-(self.C*self.x_hat_aud[t])-(self.L_del*self.x_s_aud[t])
            self.x_s_aud[t+delta_t_s_aud]=self.y_tilde_aud[t]
            self.x_hat_aud[t+delta_t_s_aud+1]=self.L1*self.x_s_aud[t+delta_t_s_aud]+self.A*self.x_hat_aud[t+delta_t_s_aud]+self.B*self.x_a_aud[t+delta_t_s_aud]+self.B*self.Kf*self.r[t+delta_t_s_aud]
            self.x_a_aud[t+delta_t_s_aud+1+delta_t_a]=(-self.K2*self.x_hat_aud[t+delta_t_s_aud+1])-self.K_del*self.x_a_aud[t+delta_t_s_aud+1]+((self.Kf-self.K4)*self.r[t+delta_t_s_aud+1])
            ##Somatosensory Sensor
            self.y_som[t] = self.y[t]+self.v_som[t] #Placeholder
            self.y_tilde_som[t]=(self.y[t])-(self.C*self.x_hat_som[t])-(self.L_del*self.x_s_som[t])
            self.x_s_som[t+delta_t_s_som]=self.y_tilde_som[t]
            self.x_hat_som[t+delta_t_s_som+1]=self.L1*self.x_s_som[t+delta_t_s_som]+self.A*self.x_hat_som[t+delta_t_s_som]+self.B*self.x_a_som[t+delta_t_s_som]+self.B*self.Kf*self.r[t+delta_t_s_som]
            self.x_a_som[t+delta_t_s_som+1+delta_t_a]=(-self.K2*self.x_hat_som[t+delta_t_s_som+1])-self.K_del*self.x_a_som[t+delta_t_s_som+1]+((self.Kf-self.K4)*self.r[t+delta_t_s_som+1])
            
            self.u[t]=self.x_a_aud[t]+self.x_a_som[t]

            #World
            self.x[t+1]=self.A*self.x[t]+self.B*self.u[t]+self.w[t]
            self.y[t+1]=self.C*self.x[t+1]+self.v[t+1]
            
            

class RelEstController(ControlSystem, PlotMixin, AnalysisMixin):
    def simulate(self, delta_t_s=1, delta_t_a=1):
        if delta_t_s == 0 and delta_t_a == 0:
            self.arch_title = 'Relative Est. System, No Delays'
        else:
            self.arch_title = 'Relative Est. System, Delays (Sensor Delay: ' + str(delta_t_s) + ', Actuator Delay: ' + str(delta_t_a) + ')'
        for t in range(0,self.time_length-1-(delta_t_s+1+delta_t_a)):
            #Brain Implementation
            self.y_tilde[t]=(self.y[t])-(self.C*self.q_hat[t])-(self.C*self.r[t])-(self.L_del*self.x_s[t])
            self.x_s[t+delta_t_s]=self.y_tilde[t]
            self.q_hat[t+delta_t_s+1]=self.L1*self.x_s[t+delta_t_s]+self.A*self.q_hat[t+delta_t_s]+self.A*self.r[t+delta_t_s]+self.B*self.x_a[t+delta_t_s]+self.B*self.Kf*self.r[t+delta_t_s]-self.r[t+delta_t_s+1]
            self.x_a[t+delta_t_s+1+delta_t_a]=(-self.K2*self.q_hat[t+delta_t_s+1])-self.K_del*self.x_a[t+delta_t_s+1]+((self.Kf-self.K4)*self.r[t+delta_t_s+1])
            self.u[t+delta_t_s+1+delta_t_a]=self.x_a[t+delta_t_s+1+delta_t_a]

            #World
            self.x[t+1]=self.A*self.x[t]+self.B*self.u[t]+self.w[t]
            self.y[t+1]=self.C*self.x[t+1]+self.v[t+1]
    def simulate_with_2sensors(self, delta_t_s_aud=1, delta_t_s_som=1, delta_t_a=1):
        if delta_t_s_aud == 0 and delta_t_s_som == 0 and delta_t_a == 0:
            self.arch_title = 'Relative Est. 2-Sensor System, No Delays'
        else:
            self.arch_title = '2-Sensor System, Delays (Auditory Sensor Delay: ' + str(delta_t_s_aud) + ', Somatosensory Sensor Delay: ' + str(delta_t_s_som) + ', Actuator Delay: ' + str(delta_t_a) + ')'
        for t in range(0,self.time_length-1-(delta_t_s_aud+1+delta_t_s_som+1+delta_t_a)):
            #Brain Implementation
            ##Auditory Sensor
            self.y_aud[t] = self.y[t]+self.v_aud[t] #Placeholder
            self.y_tilde_aud[t]=(self.y[t])-(self.C*self.q_hat_aud[t])-(self.C*self.r[t])-(self.L_del*self.x_s_aud[t])
            self.x_s_aud[t+delta_t_s_aud]=self.y_tilde_aud[t]
            self.q_hat_aud[t+delta_t_s_aud+1]=self.L1*self.x_s_aud[t+delta_t_s_aud]+self.A*self.q_hat_aud[t+delta_t_s_aud]+self.A*self.r[t+delta_t_s_aud]+self.B*self.x_a_aud[t+delta_t_s_aud]+self.B*self.Kf*self.r[t+delta_t_s_aud]-self.r[t+delta_t_s_aud+1]
            self.x_a_aud[t+delta_t_s_aud+1+delta_t_a]=(-self.K2*self.q_hat_aud[t+delta_t_s_aud+1])-self.K_del*self.x_a_aud[t+delta_t_s_aud+1]+((self.Kf-self.K4)*self.r[t+delta_t_s_aud+1])
            ##Somatosensory Sensor
            self.y_som[t] = self.y[t]+self.v_som[t] #Placeholder
            self.y_tilde_som[t]=(self.y[t])-(self.C*self.q_hat_som[t])-(self.C*self.r[t])-(self.L_del*self.x_s_som[t])
            self.x_s_som[t+delta_t_s_som]=self.y_tilde_som[t]
            self.q_hat_som[t+delta_t_s_som+1]=self.L1*self.x_s_som[t+delta_t_s_som]+self.A*self.q_hat_som[t+delta_t_s_som]+self.A*self.r[t+delta_t_s_som]+self.B*self.x_a_som[t+delta_t_s_som]+self.B*self.Kf*self.r[t+delta_t_s_som]-self.r[t+delta_t_s_som+1]
            self.x_a_som[t+delta_t_s_som+1+delta_t_a]=(-self.K2*self.q_hat_som[t+delta_t_s_som+1])-self.K_del*self.x_a_som[t+delta_t_s_som+1]+((self.Kf-self.K4)*self.r[t+delta_t_s_som+1])

            self.u[t]=self.x_a_aud[t]+self.x_a_som[t]

            #World
            self.x[t+1]=self.A*self.x[t]+self.B*self.u[t]+self.w[t]
            self.y[t+1]=self.C*self.x[t+1]+self.v[t+1]


# Simulation parameters
A = np.array([0.5])
B = np.array([1])
C = np.array([1])

# Run simulation
ref_types = ['sin'] #'sin' or 'null'
archs = ['q_based'] #['x_based','q_based']
actuator_delays = 0
sensor_delays = 3

for ref_item in ref_types:
    print('Running simulation with reference type:', ref_item)
    for arch_item in archs:
        print('Running simulation with architecture:', arch_item)
        if arch_item == 'x_based':
            system = AbsEstController(A, B, C, ref_item)    
        elif arch_item == 'q_based':
            system = RelEstController(A, B, C, ref_item)
        else:
            raise ValueError('Invalid architecture:', arch_item)

        system.simulate_with_2sensors()
        print(arch_item, ref_item, 'RMSE:', system.rmse())
        system.plot_transient(arch_item, custom_text='Error (RMSE): ' + str(system.rmse())) 