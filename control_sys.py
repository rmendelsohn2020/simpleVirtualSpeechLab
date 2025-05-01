##Control system generation

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import control as ct

dt = 0.01
T = 4 #seconds

timeseires=np.arange(0,T,dt)  
time_length = len(timeseires)

y_tilde = np.zeros((time_length,1))
print('y_tilde shape:', y_tilde.shape)
x_hat = np.zeros((time_length,1))
u = np.zeros((time_length,1))
y = np.zeros((time_length,1))
x = np.zeros((time_length,1))
w = np.random.normal(0, 0.1, (time_length,1)) # process noise

r = np.sin(timeseires)


A = np.array([1])
B = np.array([1])
C = np.array([1])

#Calculate control and sensor gains
#TODO: Calculated gains give worse result than 1?
Q=np.eye(np.shape(x[0])[0])  #TODO: Check dimensions
R=np.eye(np.shape(u[0])[0])

# Calculate the LQR gain
K, S, E = ct.dlqr(A, B, Q, R)

# Calculate the Kalman gain
G=np.eye(np.shape(w[0])[0])
QN = np.eye(np.shape(x[0])[0])
RN = np.eye(np.shape(y[0])[0])

QN=1
RN=1

L, P, E = ct.dlqe(A, G, C, QN, RN)


K1=K
K2=K
Kf=K

for t in range(0,int(T/dt)-1):
    #Brain Implementation
    y_tilde[t]=y[t]-C*x_hat[t]
    x_hat[t+1] = (L*y_tilde[t])+(A-B*K1)*x_hat[t]+(B*Kf)*r[t]
    u[t+1]=(-K2*x_hat[t+1])+(Kf*r[t+1])

    #World
    x[t+1]=A*x[t]+B*u[t]+w[t]
    y[t+1]=C*x[t+1]
    
plt.figure(figsize=(10, 6))
plt.title('Control System Simulation \n K=' + str(K) + ' Kf=' + str(Kf) + ' L=' + str(L))
plt.xlabel('Time (s)')
plt.ylabel('Values')

plt.plot(timeseires, x, label='x')
plt.plot(timeseires, y_tilde, label='y_tilde', linestyle='--')
plt.plot(timeseires, x_hat, label='x_hat')
plt.plot(timeseires, u, label='u')
plt.plot(timeseires, r, label='r')
plt.plot(timeseires, w, label='w')

plt.legend()
plt.show()
