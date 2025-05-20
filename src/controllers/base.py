import numpy as np
import control as ct

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
            self.r = np.sin(self.timeseires).reshape(-1, 1)
        elif self.ref_type == 'null':
            self.r = np.zeros((self.time_length,1))

        #Calculate control and sensor gains
        self.Q=np.eye(np.shape(self.x[0])[0])
        tune_R=1
        self.R=tune_R*np.eye(np.shape(self.u[0])[0])

        # Calculate the LQR gain
        K, S, E = ct.dlqr(self.A, self.B, self.Q, self.R)

        # Calculate the Kalman gain
        G=np.eye(np.shape(self.w[0])[0])
        QN = np.eye(np.shape(self.x[0])[0])
        tune_RN=1
        RN = tune_RN*np.eye(np.shape(self.y[0])[0])

        L, P, E = ct.dlqe(self.A, G, self.C, QN, RN)

        #Different variable names for identical gains in different locations for clarity in knockouts and noise injections
        self.K1=K
        self.K2=K
        self.K3=K
        self.K4=K
        self.Kf=K

        self.L1=L

        #Internal feedback gains for time delay compensation
        self.L_del=0
        self.K_del=0 