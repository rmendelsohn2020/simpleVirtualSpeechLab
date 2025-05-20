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
        if 'Relative Est. System' in self.arch_title:
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