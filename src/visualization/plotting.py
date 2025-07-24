import matplotlib.pyplot as plt
from pitch_pert_calibration.pitchpert_dataprep import truncate_data

# Configuration
save_figs = True
save_figs_dir = '/Users/rachelmendelsohn/Desktop/Salk/ArticulatoryFeedback/NewVSLCodebase/Figures/'

class PlotMixin:
    def plot_all(self, arch, custom_sig=None, custom_text=None, fig_save_path=None):
        plt.title(self.arch_title + ' Control System Simulation')
        plt.xlabel('Time (s)')
        plt.ylabel('Values')
        plt.plot(self.timeseries, self.x, label='x')
        if hasattr(self, 'x_hat'):
            plt.plot(self.timeseries, self.x_hat, label='x_hat')
        if hasattr(self, 'q_hat'):
            plt.plot(self.timeseries, self.q_hat, label='q_hat')
        if hasattr(self, 'q'):
            plt.plot(self.timeseries, self.q, label='q')

        if hasattr(self, 'f_A'):
            plt.plot(self.timeseries, self.f_A, label='f_A')
        if hasattr(self, 'f_S'):
            plt.plot(self.timeseries, self.f_S, label='f_S')
        if hasattr(self, 'f_Ci'):
            plt.plot(self.timeseries, self.f_Ci, label='f_Ci')
        if hasattr(self, 'f'):
            plt.plot(self.timeseries, self.f, label='f')


        if custom_sig:
            if custom_sig == 'dist':
                if hasattr(self, 'pert_P'):
                    plt.plot(self.timeseries, self.pert_P, label='pert_P')
                elif hasattr(self, 'v_aud') and hasattr(self, 'v_som'):
                    plt.plot(self.timeseries, self.v_aud, label='v_aud')
                    plt.plot(self.timeseries, self.v_som, label='v_som')
                else:
                    raise ValueError(f"Unknown system type: {self.params_obj.system_type}")


        if custom_text:
            plt.text(0.95, 0.05, custom_text, transform=plt.gca().transAxes,
                     fontsize=12, verticalalignment='bottom', horizontalalignment='right',
                     bbox=dict(facecolor='white', alpha=0.5))
        plt.legend()

        if fig_save_path:
            #self.K_str = str(self.K1).replace('.', '_')
            filename = f"{fig_save_path}/{arch}_plot_all_{self.ref_type}.png"
            plt.savefig(filename)
            print(f"Figure saved to {filename}")

        plt.show()

    def plot_transient(self, arch, start_dist=None, custom_text=None, fig_save_path=None):
        if custom_text:
            plt.text(0.95, 0.05, custom_text, transform=plt.gca().transAxes,
                     fontsize=12, verticalalignment='bottom', horizontalalignment='right',
                     bbox=dict(facecolor='white', alpha=0.5))
        if start_dist:
            self.start_dist = start_dist
        custom_time = range(self.start_dist-1,self.start_dist+8)

        #TODO: Replace cluge
        if 'abs' in arch:
            if '2sens' in arch:
                self.arch_title = 'Absolute Est. 2-Sensor System'
            else:
                self.arch_title = 'Absolute Est. System'
        elif 'rel' in arch:
            if '2sens' in arch:
                self.arch_title = 'Relative Est. 2-Sensor System'
            else:
                self.arch_title = 'Relative Est. System'
        else:
            raise ValueError(f"Unknown system type: {arch}")

        # Create figure and subplots based on system type
        if '2-Sensor System' in self.arch_title:
            # Define pairs of signals to plot side by side
            if 'Relative Est.' in self.arch_title:
                signal_pairs = [
                    (self.w, self.v, 'w', 'v'),
                    (self.v_aud, self.v_som, 'v_aud', 'v_som'),
                    (self.y_aud, self.y_som, 'y_aud', 'y_som'),
                    (self.y_tilde_aud, self.y_tilde_som, 'y_tilde_aud', 'y_tilde_som'),
                    (self.x_s_aud, self.x_s_som, 'x_s_aud', 'x_s_som'),
                    (self.q_hat_aud, self.q_hat_som, 'q_hat_aud', 'q_hat_som'),
                    (self.x_a_aud, self.x_a_som, 'x_a_aud', 'x_a_som'),
                    (self.u, self.x, 'u', 'x')
                ]
            else:
                signal_pairs = [
                    (self.w, self.v, 'w', 'v'),
                    (self.v_aud, self.v_som, 'v_aud', 'v_som'),
                    (self.y_aud, self.y_som, 'y_aud', 'y_som'),
                    (self.y_tilde_aud, self.y_tilde_som, 'y_tilde_aud', 'y_tilde_som'),
                    (self.x_s_aud, self.x_s_som, 'x_s_aud', 'x_s_som'),
                    (self.x_hat_aud, self.x_hat_som, 'x_hat_aud', 'x_hat_som'),
                    (self.x_a_aud, self.x_a_som, 'x_a_aud', 'x_a_som'),
                    (self.u, self.x, 'u', 'x')
                ]
            # Create subplots with 2 columns
            n_rows = len(signal_pairs)
            fig, axs = plt.subplots(n_rows, 2, figsize=(12, 1*n_rows), sharex=True)
            
            # Plot each pair of signals
            for i, (data1, data2, label1, label2) in enumerate(signal_pairs):
                axs[i, 0].plot(self.timeseries[custom_time], data1[custom_time], label=label1)
                axs[i, 0].set_ylabel(label1)
                axs[i, 0].legend(loc='upper right')
                
                axs[i, 1].plot(self.timeseries[custom_time], data2[custom_time], label=label2)
                axs[i, 1].set_ylabel(label2)
                axs[i, 1].legend(loc='upper right')
                
                # Add column labels
                if i == 0:
                    axs[i, 0].set_title('Auditory')
                    axs[i, 1].set_title('Somatosensory')
        else:
            if 'Absolute Est. System' in self.arch_title:
                plot_list = [self.v, self.y, self.y_tilde, self.x_s, self.x_hat, self.x_a, self.u, self.x]
                plot_labels = ['v', 'y', 'y_tilde', 'x_s', 'x_hat', 'x_a', 'u', 'x']
            elif 'Relative Est. System' in self.arch_title:
                plot_list = [self.v, self.y, self.y_tilde, self.x_s, self.q_hat, self.x_a, self.u, self.x]
                plot_labels = ['v', 'y', 'y_tilde', 'x_s', 'q_hat', 'x_a', 'u', 'x']
            else:
                raise ValueError(f"Unknown system type: {self.arch_title}")
            
            fig, axs = plt.subplots(len(plot_list), 1, figsize=(10, 8), sharex=True)
            
            for i, (data, label) in enumerate(zip(plot_list, plot_labels)):
                axs[i].plot(self.timeseries[custom_time], data[custom_time], label=label)
                axs[i].set_ylabel(label)
                axs[i].legend(loc='upper right')

        # Set the title for all cases
        fig.suptitle(self.arch_title + ' Control System Simulation\n K=' + str(self.K1) + ' Kf=' + str(self.Kf) + ' L=' + str(self.L1))

        if fig_save_path:
            self.K_str = str(self.K1).replace('.', '_')
            filename = f"{fig_save_path}/{arch}_plot_transient_{self.ref_type}_K{self.K_str}.png"
            plt.savefig(filename)
            print(f"Figure saved to {filename}")

        plt.show()

    def plot_compare_performance(self, arch, custom_text=None, fig_save_path=None):
        plt.title(self.arch_title + ' Control System Simulation\n K=' + str(self.K1) + ' Kf=' + str(self.Kf) + ' L=' + str(self.L1))
        plt.xlabel('Time (s)')
        plt.ylabel('Values')
        plt.plot(self.timeseries, self.x, label='x')
        plt.plot(self.timeseries, self.y, label='y', linestyle='--')
        plt.plot(self.timeseries, self.y_tilde, label='y_tilde', linestyle='--')
        plt.plot(self.timeseries, self.r, label='r')

        if custom_text:
            plt.text(0.95, 0.05, custom_text, transform=plt.gca().transAxes,
                     fontsize=12, verticalalignment='bottom', horizontalalignment='right',
                     bbox=dict(facecolor='white', alpha=0.5))

        plt.legend()

        if fig_save_path:
            self.K_str = str(self.K1).replace('.', '_')
            filename = f"{fig_save_path}/{arch}_compare_performance_{self.ref_type}_K{self.K_str}.png"
            plt.savefig(filename)
            print(f"Figure saved to {filename}")

        plt.show()
    def plot_signal(self, signal, label):
        plt.plot(self.timeseries, signal, label=label)
        plt.legend()
        plt.show()
    def plot_truncated(self, start_time, end_time):
        #signals to plot
        signals = [self.v_aud, self.v_som, self.y_aud, self.x]
        labels = ['v_aud', 'v_som', 'y_aud', 'x']

        #truncate data
        for i, signal in enumerate(signals):
            signal_trunc = truncate_data(self.timeseries, signal, start_time, end_time)
            time_trunc = truncate_data(self.timeseries, self.timeseries, start_time, end_time)
            plt.plot(time_trunc, signal_trunc, label=labels[i], linestyle='--')

        # plt.plot(self.time_trunc[start_time:end_time], self.v_aud[start_time:end_time], label='v_aud', linestyle='--')
        # plt.plot(self.timeseries[start_time:end_time], self.v_som[start_time:end_time], label='v_som', linestyle='--')
        # plt.plot(self.timeseries[start_time:end_time], self.y_aud[start_time:end_time], label='y_aud', linestyle='-')
        # plt.plot(self.timeseries[start_time:end_time], self.x[start_time:end_time], label='x', linestyle='-')

        plt.legend()
        plt.show()

    def plot_data_overlay(self, arch, target_response, pitch_pert_data, time_trunc=None, resp_trunc=None, pitch_pert_truncated=None, fig_save_path=None, output_dir=None):
        plt.plot(time_trunc, target_response, label='target', color='red')
        plt.plot(time_trunc, resp_trunc, label='response', color='blue')

        #plot the pitch pert
        plt.plot(time_trunc, pitch_pert_truncated, label='pitch pert, auditory', color='lightblue')
        plt.plot(time_trunc, pitch_pert_data, label='pitch pert data', color='pink')
     
        plt.legend()
        
        if fig_save_path or output_dir:
            # Use output_dir if provided, otherwise use fig_save_path
            save_dir = output_dir if output_dir is not None else fig_save_path
            filename = f"{save_dir}/{arch}_plot_data_overlay_{self.ref_type}.png"
            plt.savefig(filename)
            print(f"Figure saved to {filename}")

        plt.show()
   
    def plot_all_subplots(self, arch, custom_sig=None, custom_text=None, fig_save_path=None):
        # Collect signals and their labels
        signals = [('x', getattr(self, 'x', None))]
        if hasattr(self, 'x_hat'):
            signals.append(('x_hat', self.x_hat))
        if hasattr(self, 'q_hat'):
            signals.append(('q_hat', self.q_hat))
        if hasattr(self, 'q'):
            signals.append(('q', self.q))
        if hasattr(self, 'f_A'):
            signals.append(('f_A', self.f_A))
        if hasattr(self, 'f_S'):
            signals.append(('f_S', self.f_S))
        if hasattr(self, 'f_Ci'):
            signals.append(('f_Ci', self.f_Ci))
        if hasattr(self, 'f'):
            signals.append(('f', self.f))

        # Handle custom_sig
        if custom_sig == 'dist':
            if hasattr(self, 'pert_P'):
                signals.append(('pert_P', self.pert_P))
            elif hasattr(self, 'v_aud') and hasattr(self, 'v_som'):
                signals.append(('v_aud', self.v_aud))
                signals.append(('v_som', self.v_som))
            else:
                raise ValueError(f"Unknown system type: {self.params_obj.system_type}")

        n_signals = len(signals)
        fig, axs = plt.subplots(n_signals, 1, figsize=(8, 2*n_signals), sharex=True)

        if n_signals == 1:
            axs = [axs]  # Ensure axs is iterable

        for ax, (label, data) in zip(axs, signals):
            ax.plot(self.timeseries, data, label=label)
            ax.set_ylabel(label)
            ax.legend(loc='upper right')
            if custom_text:
                ax.text(0.95, 0.05, custom_text, transform=ax.transAxes,
                        fontsize=12, verticalalignment='bottom', horizontalalignment='right',
                        bbox=dict(facecolor='white', alpha=0.5))

        axs[-1].set_xlabel('Time (s)')
        fig.suptitle(self.arch_title + ' Control System Simulation')

        if fig_save_path:
            filename = f"{fig_save_path}/{arch}_plot_all_subplots_{self.ref_type}.png"
            plt.savefig(filename)
            print(f"Figure saved to {filename}")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
   