##Control system generation

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import control as ct

import simple_functions as sf

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

        #Virtual states for delays 
        self.x_s = np.zeros_like(self.x_hat)
        self.x_a = np.zeros_like(self.x_hat)

        np.random.seed(0) 
        #self.w = np.random.normal(0, 0.1, (self.time_length,1)) # process noise
        #step disturbance into state
        self.w = np.zeros((self.time_length,1))

        self.v = np.zeros((self.time_length,1)) # measurement noise
        self.start_dist = self.time_length//2
        self.dist_timesteps = range(self.start_dist, self.start_dist+30)
        for i in self.dist_timesteps:
            self.v[i] = self.v[i]+0.1

        self.q_hat = np.zeros((self.time_length,1))
        self.q = np.zeros((self.time_length,1))

        self.ref_type = ref_type
        if self.ref_type == 'sin':
            #self.r = np.sin(self.timeseires)
            self.r = np.sin(self.timeseires).reshape(-1, 1)
        elif self.ref_type == 'null':
            self.r = np.zeros((self.time_length,1))



        #Calculate control and sensor gains
        #TODO: Calculated gains give worse result than 1?
        self.Q=np.eye(np.shape(self.x[0])[0])  #TODO: Check dimensions
        tune_R=1
        self.R=tune_R*np.eye(np.shape(self.u[0])[0])

        # Calculate the LQR gain
        K, S, E = ct.dlqr(self.A, self.B, self.Q, self.R)

        # Calculate the Kalman gain
        G=np.eye(np.shape(self.w[0])[0])
        QN = np.eye(np.shape(self.x[0])[0])
        tune_RN=1
        RN = tune_RN*np.eye(np.shape(self.y[0])[0])

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

        #Internal feedback gains for time delay compensation
        self.L_del=0
        self.K_del=0

class AnalysisMixin:
    def rmse(self):
        sim_dif = self.x - self.r
        sim_dif_sq = sum(sim_dif**2)
        sim_mse = sim_dif_sq / self.time_length
        sim_rmse = np.sqrt(sim_mse)
        return sim_rmse

class PlotMixin:
    def plot_all(self, arch, custom_text=None):
        plt.title(self.arch_title + ' Control System Simulation\n K=' + str(self.K1) + ' Kf=' + str(self.Kf) + ' L=' + str(self.L1))
        plt.xlabel('Time (s)')
        plt.ylabel('Values')
        plt.plot(self.timeseires, self.x, label='x')
        # plt.plot(self.timeseires, self.y, label='y', linestyle='--')
        # plt.plot(self.timeseires, self.y_tilde, label='y_tilde', linestyle='--')
        if hasattr(self, 'x_hat'):
            plt.plot(self.timeseires, self.x_hat, label='x_hat')
        if hasattr(self, 'q_hat'):
            plt.plot(self.timeseires, self.q_hat, label='q_hat')
        if hasattr(self, 'q'):
            plt.plot(self.timeseires, self.q, label='q')

        # plt.plot(self.timeseires, self.u, label='u')
        # plt.plot(self.timeseires, self.r, label='r')
        # plt.plot(self.timeseires, self.w, label='w', marker='o', markersize=2, linestyle='None')

        if custom_text:
            plt.text(0.95, 0.05, custom_text, transform=plt.gca().transAxes,
                     fontsize=12, verticalalignment='bottom', horizontalalignment='right',
                     bbox=dict(facecolor='white', alpha=0.5))
        plt.legend()

        if save_figs:
            self.K_str = str(self.K1[0]).replace('.', '_')
            filename = f"{save_figs_dir}{arch}_plot_all_{self.ref_type}_K{self.K_str}.png"
            plt.savefig(filename)
            print(f"Figure saved to {filename}")

        plt.show()

    def plot_transient(self, arch, custom_text=None):
        # plt.title(self.arch_title + ' Control System Simulation\n K=' + str(self.K1) + ' Kf=' + str(self.Kf) + ' L=' + str(self.L1))
        # plt.xlabel('Time (s)')
        # plt.ylabel('Values')
        # plt.plot(self.timeseires[self.start_dist-1:self.start_dist+2], self.x[self.start_dist-1:self.start_dist+2], label='x')
        # if hasattr(self, 'x_hat'):
        #     plt.plot(self.timeseires[self.start_dist-1:self.start_dist+2], self.x_hat[self.start_dist-1:self.start_dist+2], label='x_hat')
        # if hasattr(self, 'q_hat'):
        #     plt.plot(self.timeseires[self.start_dist-1:self.start_dist+2], self.q_hat[self.start_dist-1:self.start_dist+2], label='q_hat')
        # plt.plot(self.timeseires[self.start_dist-1:self.start_dist+2], self.u[self.start_dist-1:self.start_dist+2], label='u')


            if custom_text:
                plt.text(0.95, 0.05, custom_text, transform=plt.gca().transAxes,
                     fontsize=12, verticalalignment='bottom', horizontalalignment='right',
                     bbox=dict(facecolor='white', alpha=0.5))
                
            custom_time = range(self.start_dist-1,self.start_dist+8)

            if self.arch_title == 'X-based':
                #plot v, y, x_hat, u, x
                plot_list = [self.v, self.y, self.x_hat, self.u, self.x]
                plot_labels = ['v', 'y', 'x_hat', 'u', 'x']
                fig, axs = plt.subplots(len(plot_list), 1, figsize=(10, 8), sharex=True)
            if  self.arch_title == 'Q-based':
                #plot v, y, q_hat, u, x
                plot_list = [self.v, self.y, self.q_hat, self.u, self.x]
                plot_labels = ['v', 'y', 'q_hat', 'u', 'x']
                fig, axs = plt.subplots(len(plot_list), 1, figsize=(10, 8), sharex=True)
            if self.arch_title == 'X-based, Delays':
                #plot v, y, y_tilde, x_s, x_hat, x_a, u, x
                plot_list = [self.v, self.y, self.y_tilde, self.x_s, self.x_hat, self.x_a, self.u, self.x]
                plot_labels = ['v', 'y', 'y_tilde', 'x_s', 'x_hat', 'x_a', 'u', 'x']
                fig, axs = plt.subplots(len(plot_list), 1, figsize=(10, 8), sharex=True)
            if self.arch_title == 'Q-based, Delays':
                #plot v, y, y_tilde, x_s, q_hat, x_a, u, x
                plot_list = [self.v, self.y, self.y_tilde, self.x_s, self.q_hat, self.x_a, self.u, self.x]
                plot_labels = ['v', 'y', 'y_tilde', 'x_s', 'q_hat', 'x_a', 'u', 'x']
                fig, axs = plt.subplots(len(plot_list), 1, figsize=(10, 8), sharex=True)
        
            for i, (data, label) in enumerate(zip(plot_list, plot_labels)):
                axs[i].plot(self.timeseires[custom_time], data[custom_time], label=label)
                axs[i].set_ylabel(label)
                axs[i].legend('upper right')

            fig.suptitle(self.arch_title + ' Control System Simulation\n K=' + str(self.K1) + ' Kf=' + str(self.Kf) + ' L=' + str(self.L1))

            if save_figs:
                self.K_str = str(self.K1[0]).replace('.', '_')
                filename = f"{save_figs_dir}{arch}_plot_transient_{self.ref_type}_K{self.K_str}.png"
                plt.savefig(filename)
                print(f"Figure saved to {filename}")

            plt.show()

    def plot_compare_performance(self, arch, custom_text=None):
        plt.title(self.arch_title + ' Control System Simulation\n K=' + str(self.K1) + ' Kf=' + str(self.Kf) + ' L=' + str(self.L1))
        plt.xlabel('Time (s)')
        plt.ylabel('Values')
        plt.plot(self.timeseires, self.x, label='x')
        plt.plot(self.timeseires, self.y, label='y', linestyle='--')
        plt.plot(self.timeseires, self.y_tilde, label='y_tilde', linestyle='--')
        plt.plot(self.timeseires, self.r, label='r')

        if custom_text:
            plt.text(0.95, 0.05, custom_text, transform=plt.gca().transAxes,
                     fontsize=12, verticalalignment='bottom', horizontalalignment='right',
                     bbox=dict(facecolor='white', alpha=0.5))

        plt.legend()

        if save_figs:
            self.K_str = str(self.K1[0]).replace('.', '_')
            filename = f"{save_figs_dir}{arch}_compare_performance_{self.ref_type}_K{self.K_str}.png"
            plt.savefig(filename)
            print(f"Figure saved to {filename}")

        plt.show()


class AbsoluteEstController(ControlSystem, PlotMixin, AnalysisMixin):
    def simulate(self):
        self.arch_title = 'X-based'
        for t in range(0,self.time_length-1):
            #Brain Implementation
            self.y_tilde[t]=self.y[t]-self.C*self.x_hat[t]
            self.x_hat[t+1] = (self.L1*self.y_tilde[t])+(self.A-self.B*self.K1)*self.x_hat[t]+(self.B*self.Kf)*self.r[t]
            self.u[t+1]=-(self.K2*self.x_hat[t+1])+(self.Kf*self.r[t+1])

            #World
            self.x[t+1]=self.A*self.x[t]+self.B*self.u[t]+self.w[t]
            self.y[t+1]=self.C*self.x[t+1]+self.v[t+1]

            pass

class RelativeEstController(ControlSystem, PlotMixin, AnalysisMixin):
    def simulate(self):
        self.arch_title = 'Q-based'
        for t in range(0,self.time_length-1):
            #Brain Implementation
            self.y_tilde[t]=(self.y[t])-(self.C*self.q_hat[t])-(self.C*self.r[t]) 
            self.q_hat[t+1] = (self.L1*self.y_tilde[t])+((self.A-self.B*self.K1)*self.q_hat[t])+((self.A-self.B*self.K3+self.B*self.Kf)*self.r[t])-(self.r[t+1]) 
            self.u[t+1]=(-self.K2*self.q_hat[t+1])+((self.Kf-self.K4)*self.r[t+1])

            #World 
            self.x[t+1]=self.A*self.x[t]+self.B*self.u[t]+self.w[t]
            self.y[t+1]=C*self.x[t+1]+self.v[t+1]

            self.q[t]=self.x[t]-self.r[t]
        pass

class AbsEstDelayedController(ControlSystem, PlotMixin, AnalysisMixin):
    def simulate(self, delta_t_s=1, delta_t_a=1):
        self.arch_title = 'X-based, Delays'
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
  
            pass

class RelEstDelayedController(ControlSystem, PlotMixin, AnalysisMixin):
    def simulate(self, delta_t_s=1, delta_t_a=1):
        self.arch_title = 'Q-based, Delays'
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
            pass



#Simulation parameters


A = np.array([0.5])
B = np.array([1])
C = np.array([1])


#Run simulation
ref_types = ['sin'] #'sin' or 'null'
archs= ['q_based_delayed'] #['x_based','q_based', 'x_based_delayed', 'q_based_delayed']
for ref_item in ref_types:
    print('Running simulation with reference type:', ref_item)
    for arch_item in archs:
        print('Running simulation with architecture:', arch_item)
        if arch_item == 'x_based':
            system = AbsoluteEstController(A, B, C, ref_item)
        elif arch_item == 'x_based_delayed':
            system = AbsEstDelayedController(A, B, C, ref_item)    
        elif arch_item == 'q_based':
            system = RelativeEstController(A, B, C, ref_item)
        elif arch_item == 'q_based_delayed':
            system = RelEstDelayedController(A, B, C, ref_item)

        system.simulate()
        print(arch_item, ref_item, 'RMSE:', system.rmse())
        #system.plot_all(arch_item, custom_text='Error (RMSE): ' + str(system.rmse()))
        system.plot_transient(arch_item, custom_text='Error (RMSE): ' + str(system.rmse()))
        #system.plot_compare_performance(arch_item, custom_text='Error (RMSE): ' + str(system.rmse()))
