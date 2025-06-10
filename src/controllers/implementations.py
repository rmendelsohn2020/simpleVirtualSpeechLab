from .base import ControlSystem
from utils.analysis import AnalysisMixin
from visualization.plotting import PlotMixin 
from utils.processing import set_array_value

class AbsEstController(ControlSystem, AnalysisMixin, PlotMixin):
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
            self.x_a[t+delta_t_s+1+delta_t_a]=-self.K1*self.x_hat[t+delta_t_s+1]-self.K_del*self.x_a[t+delta_t_s+1]+self.Kf*self.r[t+delta_t_s+1]
            self.u[t+delta_t_s+1+delta_t_a]= self.x_a[t+delta_t_s+1+delta_t_a]

            #World
            self.x[t+1]=self.A*self.x[t]+self.B*self.u[t]+self.w[t]
            self.y[t+1]=self.C*self.x[t+1]+self.v[t+1]

    def simulate_with_2sensors(self, delta_t_s_aud=1, delta_t_s_som=1, delta_t_a=1):
        if delta_t_s_aud == 0 and delta_t_s_som == 0 and delta_t_a == 0:
            self.arch_title = '2-Sensor System, No Delays'
        else:
            self.arch_title = '2-Sensor System, Delays (Auditory Sensor Delay: ' + str(delta_t_s_aud) + ', Somatosensory Sensor Delay: ' + str(delta_t_s_som) + ', Actuator Delay: ' + str(delta_t_a) + ')'
        for t in range(0,self.time_length-1-(delta_t_s_aud+1+delta_t_s_som+1+delta_t_a)):
            #Brain Implementation
            #TODO: One function for all sensors
            ##Auditory Sensor
            self.y_aud[t] = self.C*self.x[t]+self.v_aud[t]
            self.y_tilde_aud[t]=self.y_aud[t]-self.C*self.x_hat_aud[t]-self.aud_L_del*self.x_s_aud[t]
            self.x_s_aud[t+delta_t_s_aud]=self.y_tilde_aud[t]
            self.x_hat_aud[t+delta_t_s_aud+1] = (self.aud_L1*self.x_s_aud[t+delta_t_s_aud])+(self.A*self.x_hat_aud[t+delta_t_s_aud])+(self.B*self.x_a_aud[t+delta_t_s_aud])+(self.B*self.aud_Kf)*self.r[t+delta_t_s_aud]
            self.x_a_aud[t+delta_t_s_aud+1+delta_t_a]=-self.aud_K1*self.x_hat_aud[t+delta_t_s_aud+1]-self.aud_K_del*self.x_a_aud[t+delta_t_s_aud+1]+self.aud_Kf*self.r[t+delta_t_s_aud+1]
            
            ##Somatosensory Sensor
            self.y_som[t] = self.y[t]+self.v_som[t]
            self.y_tilde_som[t]=self.y_som[t]-self.C*self.x_hat_som[t]-self.som_L_del*self.x_s_som[t]
            self.x_s_som[t+delta_t_s_som]=self.y_tilde_som[t]
            self.x_hat_som[t+delta_t_s_som+1] = (self.som_L1*self.x_s_som[t+delta_t_s_som])+(self.A*self.x_hat_som[t+delta_t_s_som])+(self.B*self.x_a_som[t+delta_t_s_som])+(self.B*self.som_Kf)*self.r[t+delta_t_s_som]
            self.x_a_som[t+delta_t_s_som+1+delta_t_a]=-self.som_K1*self.x_hat_som[t+delta_t_s_som+1]-self.som_K_del*self.x_a_som[t+delta_t_s_som+1]+self.som_Kf*self.r[t+delta_t_s_som+1]

            self.u[t]=self.x_a_aud[t]+self.x_a_som[t]

            #World
            self.x[t+1]=self.A*self.x[t]+self.B*self.u[t]+self.w[t]
            self.y[t+1]=self.C*self.x[t+1]+self.v[t+1]
            


class RelEstController(ControlSystem, AnalysisMixin, PlotMixin):
    def simulate_with_1sensor(self, delta_t_s=1, delta_t_a=1):
        # Initialize sensor delay attributes
        self.delta_t_s = delta_t_s
        self.delta_t_a = delta_t_a

        if self.delta_t_s == 0 and self.delta_t_a == 0:
            self.arch_title = 'Relative Est. System, No Delays'
        else:
            self.arch_title = 'Relative Est. System, Delays (Sensor Delay: ' + str(self.delta_t_s) + ', Actuator Delay: ' + str(self.delta_t_a) + ')'
        for t in range(0,self.time_length-1-(self.delta_t_s+1+self.delta_t_a)):
            #Brain Implementation
            self.process_sensor_channel(t, self.delta_t_s, self.delta_t_a, channel=None)

            #World/Non-sensor specific processing
            self.process_global(t, channels=None)

    def simulate_with_2sensors(self, delta_t_s_aud=1, delta_t_s_som=1, delta_t_a=1):
        # Initialize sensor delay attributes
        self.delta_t_s_aud = delta_t_s_aud
        self.delta_t_s_som = delta_t_s_som
        self.delta_t_a = delta_t_a

        #Set architecture title
        if self.delta_t_s_aud == 0 and self.delta_t_s_som == 0 and self.delta_t_a == 0:
            self.arch_title = 'Relative Est. 2-Sensor System, No Delays'
        else:
            self.arch_title = 'Relative Est. 2-Sensor System, Delays (Auditory Sensor Delay: ' + str(self.delta_t_s_aud) + ', Somatosensory Sensor Delay: ' + str(self.delta_t_s_som) + ', Actuator Delay: ' + str(self.delta_t_a) + ')'
     
        #Process each sensor channel
        for t in range(0,self.time_length-1-(self.delta_t_s_aud+1+self.delta_t_s_som+1+self.delta_t_a)):
            #Brain Implementation
            for channel in ["aud", "som"]:
                delta_t_s_ch = getattr(self, f"delta_t_s_{channel}")
                self.process_sensor_channel(t, delta_t_s_ch, self.delta_t_a, channel)
             
            #World/Non-sensor specific processing
            channels = ["aud", "som"]
            self.process_global(t, channels)


    def process_sensor_channel(self, t, delta_t_s, delta_t_a, channel):
        if channel == "aud" or channel == "som":
            channel_prefix = f"{channel}_"
            channel_suffix = f"_{channel}"
        else:
            channel_prefix = ""
            channel_suffix = ""

        # Get signals
        y = getattr(self, f"y{channel_suffix}")
        y_tilde = getattr(self, f"y_tilde{channel_suffix}")
        q_hat = getattr(self, f"q_hat{channel_suffix}")
        x_s = getattr(self, f"x_s{channel_suffix}")
        x_a = getattr(self, f"x_a{channel_suffix}")
        v = getattr(self, f"v{channel_suffix}")

        # Get parameters
        C = self.C_aud if channel == "aud" else self.C_som if channel == "som" else self.C
        L1 = getattr(self, f"{channel_prefix}L1")
        L_del = getattr(self, f"{channel_prefix}L_del")
        Kf1 = getattr(self, f"{channel_prefix}Kf1")
        Kf2 = getattr(self, f"{channel_prefix}Kf2")
        K1 = getattr(self, f"{channel_prefix}K1")
        K2 = getattr(self, f"{channel_prefix}K2")
        K_del = getattr(self, f"{channel_prefix}K_del")


        # Compute y_tilde
        y_tilde_val = (y[t]) - (C * q_hat[t]) - (C * self.r[t]) - (L_del * x_s[t])
        time_index = t
        setattr(self, f"y_tilde{channel_suffix}", set_array_value(y_tilde, time_index, y_tilde_val))

        # Update x_s
        x_s_val = y_tilde_val
        time_index = t + delta_t_s
        setattr(self, f"x_s{channel_suffix}", set_array_value(x_s, time_index, x_s_val))

        # Update x_hat
        q_hat_val = (L1 * x_s[t + delta_t_s] +
                    self.A * q_hat[t + delta_t_s] +
                    self.A * self.r[t + delta_t_s] +
                    self.B * x_a[t + delta_t_s] +
                    self.B * Kf1 * self.r[t + delta_t_s] -
                    self.r[t + delta_t_s + 1])
        time_index = t + delta_t_s + 1
        setattr(self, f"q_hat{channel_suffix}", set_array_value(q_hat, time_index, q_hat_val))

        # Update x_a
        x_a_val = (-K1 * q_hat[t + delta_t_s + 1] -
                K_del * x_a[t + delta_t_s + 1] +
                (Kf2-K2) * self.r[t + delta_t_s + 1])
        time_index = t + delta_t_s + 1 + self.delta_t_a
        setattr(self, f"x_a{channel_suffix}", set_array_value(x_a, t + delta_t_s + 1 + delta_t_a, x_a_val))
    
    def process_global(self, t, channels):
        #Sum control signals from all channels
        self.u[t]=self.x_a_aud[t]+self.x_a_som[t]

        for channel in channels:
            #Update y for each channel
            if channel == "aud" or channel == "som":
                channel_prefix = f"{channel}_"
                channel_suffix = f"_{channel}"
            else:
                channel_prefix = ""
                channel_suffix = ""
            
            v_channel = getattr(self, f"v{channel_suffix}")
            y_channel = getattr(self, f"y{channel_suffix}")
            C_channel = getattr(self, f"C{channel_suffix}")
            y_val = C_channel*self.x[t+1]+v_channel[t+1]
            setattr(self, f"y{channel_suffix}", set_array_value(y_channel, t+1, y_val))

        #Update x
        self.x[t+1]=self.A*self.x[t]+self.B*self.u[t]+self.w[t]
    