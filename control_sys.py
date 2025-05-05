##Control system generation

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import control as ct

save_figs = True
save_figs_dir = '/Users/rachelmendelsohn/Desktop/Salk/ArticulatoryFeedback/NewVSLCodebase/Figures/'


class ControlSystem:
    def __init__(self, input_A, input_B, input_C, ref_type='sin'):
       #NOTE: input arguments can describe certain features of the system matrices,
       #but currently are used to directly as the matrices A, B, C, Q and R
        self.A = input_A
        self.B = input_B
        self.C = input_C

        ###Simulation parameters
        self.dt = 0.01
        self.T = 4 #seconds

        self.timeseires=np.arange(0,self.T,self.dt)  
        self.time_length = len(self.timeseires)

        ###System parameters
        self.y_tilde = np.zeros((self.time_length,1))
        self.x_hat = np.zeros((self.time_length,1))
        self.u = np.zeros((self.time_length,1))
        self.y = np.zeros((self.time_length,1))
        self.x = np.zeros((self.time_length,1))

        np.random.seed(0) 
        self.w = np.random.normal(0, 0.1, (self.time_length,1)) # process noise


        self.q_hat = np.zeros((self.time_length,1))
        self.q = np.zeros((self.time_length,1))

        self.ref_type = ref_type
        if self.ref_type == 'sin':
            self.r = np.sin(self.timeseires)
        elif self.ref_type == 'null':
            self.r = np.zeros((self.time_length,1))


        self.delta_r = np.zeros_like(self.r)
        for i in range(1, len(self.r)):
            self.delta_r[i] = self.r[i] - self.r[i-1]
        print('delta_r shape:', self.delta_r.shape)


        #Calculate control and sensor gains
        #TODO: Calculated gains give worse result than 1?
        self.Q=np.eye(np.shape(self.x[0])[0])  #TODO: Check dimensions
        self.R=np.eye(np.shape(self.u[0])[0])

        # Calculate the LQR gain
        K, S, E = ct.dlqr(self.A, self.B, self.Q, self.R)

        # Calculate the Kalman gain
        G=np.eye(np.shape(self.w[0])[0])
        QN = np.eye(np.shape(self.x[0])[0])
        RN = np.eye(np.shape(self.y[0])[0])

        # QN=1
        # RN=1

        L, P, E = ct.dlqe(self.A, G, self.C, QN, RN)

        # #NOTE: Overwrite the gains with 1 for testing
        # K=1
        # L=1

        #Different variable names for identical gains in different locations for clarity in knockouts and noise injections
        self.K1=K
        self.K2 =K
        self.K3=K
        self.K4=K
        self.Kf=K

        self.L1=L

class PlotMixin:
    def plot_all(self, arch):
        plt.title(self.arch_title + ' Control System Simulation\n K=' + str(self.K1) + ' Kf=' + str(self.Kf) + ' L=' + str(self.L1))
        plt.xlabel('Time (s)')
        plt.ylabel('Values')
        plt.plot(self.timeseires, self.x, label='x')
        plt.plot(self.timeseires, self.y, label='y', linestyle='--')
        plt.plot(self.timeseires, self.y_tilde, label='y_tilde', linestyle='--')
        if hasattr(self, 'x_hat'):
            plt.plot(self.timeseires, self.x_hat, label='x_hat')
        if hasattr(self, 'q_hat'):
            plt.plot(self.timeseires, self.q_hat, label='q_hat')
        if hasattr(self, 'q'):
            plt.plot(self.timeseires, self.q, label='q')

        plt.plot(self.timeseires, self.u, label='u')
        plt.plot(self.timeseires, self.r, label='r')
        plt.plot(self.timeseires, self.w, label='w')

        plt.legend()

        if save_figs:
            filename = f"{save_figs_dir}{arch}_plot_all_{self.ref_type}.png"
            plt.savefig(filename)
            print(f"Figure saved to {filename}")

        plt.show()

    def plot_compare_performance(self, arch):
        plt.title(self.arch_title + ' Control System Simulation\n K=' + str(self.K1) + ' Kf=' + str(self.Kf) + ' L=' + str(self.L1))
        plt.xlabel('Time (s)')
        plt.ylabel('Values')
        plt.plot(self.timeseires, self.x, label='x')
        plt.plot(self.timeseires, self.y, label='y', linestyle='--')
        plt.plot(self.timeseires, self.y_tilde, label='y_tilde', linestyle='--')
        plt.plot(self.timeseires, self.r, label='r')

        plt.legend()

        if save_figs:
            filename = f"{save_figs_dir}{arch}_compare_performance_{self.ref_type}.png"
            plt.savefig(filename)
            print(f"Figure saved to {filename}")

        plt.show()


class AbsoluteEstController(ControlSystem, PlotMixin):
    def simulate(self):
        self.arch_title = 'X-based'
        for t in range(0,self.time_length-1):
            #Brain Implementation
            self.y_tilde[t]=self.y[t]-self.C*self.x_hat[t]
            self.x_hat[t+1] = (self.L1*self.y_tilde[t])+(self.A-self.B*self.K1)*self.x_hat[t]+(self.B*self.Kf)*self.r[t]
            self.u[t+1]=(-self.K2*self.x_hat[t+1])+(self.Kf*self.r[t+1])

            #World
            self.x[t+1]=self.A*self.x[t]+self.B*self.u[t]+self.w[t]
            self.y[t+1]=self.C*self.x[t+1]

            pass

class RelativeEstController(ControlSystem, PlotMixin):
    def simulate(self):
        self.arch_title = 'Q-based'
        for t in range(0,self.time_length-1):
            #Brain Implementation
            self.y_tilde[t]=(self.y[t])-(self.C*self.q_hat[t])-(self.C*self.r[t]) 
            self.q_hat[t+1] = (self.L1*self.y_tilde[t])+((self.A-self.B*self.K1)*self.q_hat[t])+((self.A-self.B*self.K3+self.B*self.Kf)*self.r[t])-(self.r[t+1]) 
            self.u[t+1]=(-self.K2*self.q_hat[t+1])+((self.Kf-self.K4)*self.r[t+1])

            #World 
            self.x[t+1]=self.A*self.x[t]+self.B*self.u[t]+self.w[t]
            self.y[t+1]=C*self.x[t+1]

            self.q[t]=self.x[t]-self.r[t]
        pass




#Simulation parameters
ref_type = 'null' #'sin' or 'null'

A = np.array([1])
B = np.array([1])
C = np.array([1])


#Run simulation
arch= 'q_based' # or 'q_based'
if arch == 'x_based':
    system = AbsoluteEstController(A, B, C, ref_type)
elif arch == 'q_based':
    system = RelativeEstController(A, B, C, ref_type)

system.simulate()
system.plot_all(arch)
system.plot_compare_performance(arch)