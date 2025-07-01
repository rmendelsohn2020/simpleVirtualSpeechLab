from abc import ABC, abstractmethod
from .base import ControlSystem
from utils.analysis import AnalysisMixin
from visualization.plotting import PlotMixin 
from utils.processing import set_array_value

class SensorProcessor(ABC):
    """Abstract base class defining the interface for sensor processing strategies"""
    @abstractmethod
    def process_sensor_channel(self, controller, t, delta_t_s, delta_t_a, channel):
        """
        Process a single sensor channel at time t
        
        Args:
            controller: The controller instance containing state variables
            t: Current time step
            delta_t_s: Sensor delay
            delta_t_a: Actuator delay
            channel: Channel identifier ("aud", "som", or None)
        """
        pass

class AbsoluteSensorProcessor(SensorProcessor):
    """Concrete implementation for absolute estimation processing"""
    def process_sensor_channel(self, controller, t, delta_t_s, delta_t_a, channel):
        controller.arch_str = "Absolute Est."
        if channel == "aud" or channel == "som":
            channel_prefix = f"{channel}_"
            channel_suffix = f"_{channel}"
        else:
            channel_prefix = ""
            channel_suffix = ""

        # Get signals
        y = getattr(controller, f"y{channel_suffix}")
        y_tilde = getattr(controller, f"y_tilde{channel_suffix}")
        x_hat = getattr(controller, f"x_hat{channel_suffix}")
        x_s = getattr(controller, f"x_s{channel_suffix}")
        x_a = getattr(controller, f"x_a{channel_suffix}")

        # Get parameters
        C = controller.C_aud if channel == "aud" else controller.C_som if channel == "som" else controller.C
        L1 = getattr(controller, f"{channel_prefix}L1")
        L_del = getattr(controller, f"{channel_prefix}L_del")
        K1 = getattr(controller, f"{channel_prefix}K1")
        K_del = getattr(controller, f"{channel_prefix}K_del")
        Kf = getattr(controller, f"{channel_prefix}Kf")

        # Compute y_tilde
        y_tilde_val = y[t] - C * x_hat[t] - L_del * x_s[t]
        time_index = t
        setattr(controller, f"y_tilde{channel_suffix}", set_array_value(y_tilde, time_index, y_tilde_val))

        # Update x_s
        x_s_val = y_tilde_val
        time_index = t + delta_t_s
        setattr(controller, f"x_s{channel_suffix}", set_array_value(x_s, time_index, x_s_val))

        # Update x_hat
        x_hat_val = (L1 * x_s[t + delta_t_s] +
                    controller.A * x_hat[t + delta_t_s] +
                    controller.B * x_a[t + delta_t_s] +
                    controller.B * Kf * controller.r[t + delta_t_s])
        time_index = t + delta_t_s + 1
        setattr(controller, f"x_hat{channel_suffix}", set_array_value(x_hat, time_index, x_hat_val))

        # Update x_a
        x_a_val = (-K1 * x_hat[t + delta_t_s + 1] -
                  K_del * x_a[t + delta_t_s + 1] +
                  Kf * controller.r[t + delta_t_s + 1])
        time_index = t + delta_t_s + 1 + delta_t_a
        setattr(controller, f"x_a{channel_suffix}", set_array_value(x_a, time_index, x_a_val))

class RelativeSensorProcessor(SensorProcessor):
    """Concrete implementation for relative estimation processing"""
    def process_sensor_channel(self, controller, t, delta_t_s, delta_t_a, channel):
        controller.arch_str = "Relative Est."
        if channel == "aud" or channel == "som":
            channel_prefix = f"{channel}_"
            channel_suffix = f"_{channel}"
        else:
            channel_prefix = ""
            channel_suffix = ""

        # Get signals
        y = getattr(controller, f"y{channel_suffix}")
        y_tilde = getattr(controller, f"y_tilde{channel_suffix}")
        q_hat = getattr(controller, f"q_hat{channel_suffix}")
        x_s = getattr(controller, f"x_s{channel_suffix}")
        x_a = getattr(controller, f"x_a{channel_suffix}")

        # Get parameters
        C = controller.C_aud if channel == "aud" else controller.C_som if channel == "som" else controller.C
        L1 = getattr(controller, f"{channel_prefix}L1")
        L_del = getattr(controller, f"{channel_prefix}L_del")
        Kf1 = getattr(controller, f"{channel_prefix}Kf1")
        Kf2 = getattr(controller, f"{channel_prefix}Kf2")
        K1 = getattr(controller, f"{channel_prefix}K1")
        K2 = getattr(controller, f"{channel_prefix}K2")
        K_del = getattr(controller, f"{channel_prefix}K_del")

        # Compute y_tilde
        y_tilde_val = (y[t]) - (C * q_hat[t]) - (C * controller.r[t]) - (L_del * x_s[t])
        time_index = t
        setattr(controller, f"y_tilde{channel_suffix}", set_array_value(y_tilde, time_index, y_tilde_val))

        # Update x_s
        x_s_val = y_tilde_val
        time_index = t + delta_t_s
        setattr(controller, f"x_s{channel_suffix}", set_array_value(x_s, time_index, x_s_val))

        # Update x_hat
        q_hat_val = (L1 * x_s[t + delta_t_s] +
                    controller.A * q_hat[t + delta_t_s] +
                    controller.A * controller.r[t + delta_t_s] +
                    controller.B * x_a[t + delta_t_s] +
                    controller.B * Kf1 * controller.r[t + delta_t_s] -
                    controller.r[t + delta_t_s + 1])
        time_index = t + delta_t_s + 1
        setattr(controller, f"q_hat{channel_suffix}", set_array_value(q_hat, time_index, q_hat_val))

        # Update x_a
        x_a_val = (-K1 * q_hat[t + delta_t_s + 1] -
                  K_del * x_a[t + delta_t_s + 1] +
                  (Kf2-K2) * controller.r[t + delta_t_s + 1])
        time_index = t + delta_t_s + 1 + delta_t_a
        setattr(controller, f"x_a{channel_suffix}", set_array_value(x_a, time_index, x_a_val))

class Controller(ControlSystem, AnalysisMixin, PlotMixin):
    """Base controller class that uses a sensor processor strategy"""
    def __init__(self, sensor_processor: SensorProcessor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sensor_processor = sensor_processor
        print(sensor_processor)


    def process_global(self, t, channels):
        #Sum control signals from all channels
        self.u[t] = self.x_a_aud[t] + self.x_a_som[t]

        #Update x
        self.x[t+1] = self.A*self.x[t] + self.B*self.u[t] + self.w[t]
        
        if channels is None:
            channels = [""]
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
        # self.x[t+1] = self.A*self.x[t] + self.B*self.u[t] + self.w[t]
        
    def simulate_with_1sensor(self, delta_t_s=1, delta_t_a=1):
        self.delta_t_s = delta_t_s
        self.delta_t_a = delta_t_a

        # if self.delta_t_s == 0 and self.delta_t_a == 0:
        #     self.arch_title = f' System, No Delays'
        # else:
        #     self.arch_title = f' System, Delays (Sensor Delay: {self.delta_t_s}, Actuator Delay: {self.delta_t_a})'

        for t in range(0, self.time_length-1-(self.delta_t_s+1+self.delta_t_a)):
            self.sensor_processor.process_sensor_channel(self, t, self.delta_t_s, self.delta_t_a, None)
            self.process_global(t, channels=None)
        if self.delta_t_s == 0 and self.delta_t_a == 0:
            self.arch_title = f' {self.arch_str} 1-Sensor System, No Delays'
        else:
            self.arch_title = f' {self.arch_str} 1-Sensor System, Delays (Sensor Delay: {self.delta_t_s}, Actuator Delay: {self.delta_t_a})'

    def simulate_with_2sensors(self, delta_t_s_aud=1, delta_t_s_som=1, delta_t_a=1):
        self.delta_t_s_aud = delta_t_s_aud
        self.delta_t_s_som = delta_t_s_som
        self.delta_t_a = delta_t_a

        
        for t in range(0, self.time_length-1-(self.delta_t_s_aud+1+self.delta_t_s_som+1+self.delta_t_a)):
            for channel in ["aud", "som"]:
                delta_t_s_ch = getattr(self, f"delta_t_s_{channel}")
                self.sensor_processor.process_sensor_channel(self, t, delta_t_s_ch, self.delta_t_a, channel)
            
            self.process_global(t, channels=["aud", "som"])

        if self.delta_t_s_aud == 0 and self.delta_t_s_som == 0 and self.delta_t_a == 0:
            self.arch_title = f' {self.arch_str} 2-Sensor System, No Delays'
        else:
            self.arch_title = f' {self.arch_str} 2-Sensor System, Delays (Auditory Sensor Delay: {self.delta_t_s_aud}, Somatosensory Sensor Delay: {self.delta_t_s_som}, Actuator Delay: {self.delta_t_a})'
