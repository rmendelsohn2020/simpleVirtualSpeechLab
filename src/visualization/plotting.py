import matplotlib.pyplot as plt

# Configuration
save_figs = True
save_figs_dir = '/Users/rachelmendelsohn/Desktop/Salk/ArticulatoryFeedback/NewVSLCodebase/Figures/'

class PlotMixin:
    def plot_all(self, arch, custom_text=None):
        plt.title(self.arch_title + ' Control System Simulation\n K=' + str(self.K1) + ' Kf=' + str(self.Kf) + ' L=' + str(self.L1))
        plt.xlabel('Time (s)')
        plt.ylabel('Values')
        plt.plot(self.timeseires, self.x, label='x')
        if hasattr(self, 'x_hat'):
            plt.plot(self.timeseires, self.x_hat, label='x_hat')
        if hasattr(self, 'q_hat'):
            plt.plot(self.timeseires, self.q_hat, label='q_hat')
        if hasattr(self, 'q'):
            plt.plot(self.timeseires, self.q, label='q')

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
        if custom_text:
            plt.text(0.95, 0.05, custom_text, transform=plt.gca().transAxes,
                     fontsize=12, verticalalignment='bottom', horizontalalignment='right',
                     bbox=dict(facecolor='white', alpha=0.5))
                
        custom_time = range(self.start_dist-1,self.start_dist+8)
        if 'Absolute Est. System' in self.arch_title:
            plot_list = [self.v, self.y, self.y_tilde, self.x_s, self.x_hat, self.x_a, self.u, self.x]
            plot_labels = ['v', 'y', 'y_tilde', 'x_s', 'x_hat', 'x_a', 'u', 'x']
            fig, axs = plt.subplots(len(plot_list), 1, figsize=(10, 8), sharex=True)
        elif 'Relative Est. System' in self.arch_title:
            plot_list = [self.v, self.y, self.y_tilde, self.x_s, self.q_hat, self.x_a, self.u, self.x]
            plot_labels = ['v', 'y', 'y_tilde', 'x_s', 'q_hat', 'x_a', 'u', 'x']
            fig, axs = plt.subplots(len(plot_list), 1, figsize=(10, 8), sharex=True)
        elif '2-Sensor System' in self.arch_title:
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
                axs[i, 0].plot(self.timeseires[custom_time], data1[custom_time], label=label1)
                axs[i, 0].set_ylabel(label1)
                axs[i, 0].legend(loc='upper right')
                
                axs[i, 1].plot(self.timeseires[custom_time], data2[custom_time], label=label2)
                axs[i, 1].set_ylabel(label2)
                axs[i, 1].legend(loc='upper right')
                
                # Add column labels
                if i == 0:
                    axs[i, 0].set_title('Auditory')
                    axs[i, 1].set_title('Somatosensory')
            
            # Adjust layout
            plt.tight_layout()
            
        else:
            raise ValueError(f"Unknown system type: {self.arch_title}")
    
        if '2-Sensor System' not in self.arch_title:
            for i, (data, label) in enumerate(zip(plot_list, plot_labels)):
                axs[i].plot(self.timeseires[custom_time], data[custom_time], label=label)
                axs[i].set_ylabel(label)
                axs[i].legend(loc='upper right')

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