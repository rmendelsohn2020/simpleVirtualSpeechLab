import numpy as np
import control as ct

class ControlSystem:
    def __init__(self, input_A, input_B, input_C, ref_type='sin', dist_custom=None, dist_type=['State'], timeseries=None, tune_R=1, tune_RN=1):
        #NOTE: input arguments can describe certain features of the system matrices,
        #but currently are used to directly as the matrices A, B, C, Q and R
        self.A = input_A
        self.B = input_B
        self.C = input_C

        ###Simulation parameters
        if timeseries is not None:
            self.timeseries = timeseries
            self.time_length = len(self.timeseries)
        else:    
            self.dt = 0.01
            self.T = 4 #seconds

            self.timeseries=np.arange(0,self.T,self.dt)  
            self.time_length = len(self.timeseries)

        ###System parameters
        self.y_tilde = np.zeros((self.time_length,1))
        self.x_hat = np.zeros((self.time_length,1))
        self.q_hat = np.zeros((self.time_length,1))
        self.u = np.zeros((self.time_length,1))
        self.y = np.zeros((self.time_length,1))
        self.x = np.zeros((self.time_length,1))

        #Virtual states for delays 
        self.x_s = np.zeros_like(self.x_hat)
        self.x_a = np.zeros_like(self.x_hat)

        #Sensory channel states
        ##Somatosensory
        self.y_som = np.zeros_like(self.y)
        self.y_tilde_som = np.zeros_like(self.y)
        self.x_s_som = np.zeros_like(self.x_s)
        self.x_a_som = np.zeros_like(self.x_a)
        self.x_hat_som = np.zeros_like(self.x_hat)
        self.q_hat_som = np.zeros_like(self.q_hat)
        ##Auditory
        self.y_aud = np.zeros_like(self.y)
        self.y_tilde_aud = np.zeros_like(self.y)
        self.x_s_aud = np.zeros_like(self.x_s)
        self.x_a_aud = np.zeros_like(self.x_a)
        self.x_hat_aud = np.zeros_like(self.x_hat)
        self.q_hat_aud = np.zeros_like(self.q_hat)


        np.random.seed(0) 
        #self.w = np.zeros((self.time_length,1))
        # State noise
        self.w = np.random.normal(0, 0.01, (self.time_length, 1))  # Gaussian noise with mean 0 and std 0.01

        #Measurement noise

        self.v = np.zeros((self.time_length,1)) # measurement noise
        self.v_aud = np.zeros((self.time_length,1)) # auditory measurement noise
        self.v_som = np.zeros((self.time_length,1)) # somatosensory measurement noise


        if dist_custom is not None:
            dist_sig = dist_custom
        else:
            #Default to a step change in the state at the middle of the simulation
            dist_sig = np.zeros((self.time_length,1))
            self.start_dist = self.time_length//2
            self.dist_timesteps = range(self.start_dist, self.start_dist+30)
            for i in self.dist_timesteps:
                dist_sig[i] = dist_sig[i]+0.1

        if 'Auditory' in dist_type:
            self.v_aud = dist_sig
        if 'Somatosensory' in dist_type:
            self.v_som = dist_sig
        if 'State' in dist_type:
            self.v = dist_sig


        self.q_hat = np.zeros((self.time_length,1))
        self.q = np.zeros((self.time_length,1))

        print(type(timeseries))
        self.ref_type = ref_type
        if self.ref_type == 'sin':
            self.r = np.sin(self.timeseries).reshape(-1, 1)
        elif self.ref_type == 'null':
            self.r = np.zeros((self.time_length,1))

        # Calculate gains for main controller
        self.calculate_gains(self.A, self.B, self.C)
        
        # Calculate gains for auditory and somatosensory controllers
        self.calculate_gains(self.A, self.B, self.C, tune_R=tune_R, tune_RN=tune_RN, prefix='aud_')
        self.calculate_gains(self.A, self.B, self.C, tune_R=tune_R, tune_RN=tune_RN, prefix='som_')

    def calculate_gains(self, A, B, C, tune_R=1, tune_RN=1, prefix=''):
        # Calculate control and sensor gains
        Q = np.eye(np.shape(self.x[0])[0])
        R = tune_R * np.eye(np.shape(self.u[0])[0])

        # Calculate the LQR gain
        K, S, E = ct.dlqr(A, B, Q, R)

        # Calculate the Kalman gain
        G = np.eye(np.shape(self.w[0])[0])
        QN = np.eye(np.shape(self.x[0])[0])
        RN = tune_RN * np.eye(np.shape(self.y[0])[0])

        L, P, E = ct.dlqe(A, G, C, QN, RN)

        # Set gains with optional prefix for different controllers
        setattr(self, prefix + 'K1', K)
        setattr(self, prefix + 'K2', K) 
        setattr(self, prefix + 'K3', K)
        setattr(self, prefix + 'K4', K)
        setattr(self, prefix + 'Kf', K)
        setattr(self, prefix + 'L1', L)

        # Internal feedback gains for time delay compensation
        setattr(self, prefix + 'L_del', 0)
        setattr(self, prefix + 'K_del', 0)