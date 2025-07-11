from utils.get_configs import get_paths
from datetime import datetime

def readout_optimized_params(cal_params, sensor_delay_aud=None, sensor_delay_som=None, actuator_delay=None, format_opt=['txt','print'], output_dir=None):
    if output_dir is None:
        path_obj = get_paths()
        output_dir = path_obj.fig_save_path

    if 'print' in format_opt:
        if cal_params.system_type == 'DIVA':
            print(f'Optimized DIVA ({cal_params.kearney_name}) parameters:')
            print(f'alpha_A: {cal_params.alpha_A_init}')
            print(f'alpha_S: {cal_params.alpha_S_init}')
            print(f'alpha_Av: {cal_params.alpha_Av_init}')
            print(f'alpha_Sv: {cal_params.alpha_Sv_init}')
            print(f'tau_A: {cal_params.tau_A_init}')
            print(f'tau_S: {cal_params.tau_S_init}')
            print(f'tau_As: {cal_params.tau_As_init}')
            print(f'tau_Ss: {cal_params.tau_Ss_init}')
        elif cal_params.system_type == 'Template':
            print(f'Optimized Template parameters:')
            print(f'A: {cal_params.A_init}')
            print(f'B: {cal_params.B_init}')
            print(f'C_aud: {cal_params.C_aud_init}')
            print(f'C_som: {cal_params.C_som_init}')
            print(f'K_aud: {cal_params.K_aud_init}')
            print(f'L_aud: {cal_params.L_aud_init}')
            print(f'K_som: {cal_params.K_som_init}')
            print(f'L_som: {cal_params.L_som_init}')

        print(f'sensor_delay_aud: {sensor_delay_aud}')
        print(f'sensor_delay_som: {sensor_delay_som}')
        print(f'actuator_delay: {actuator_delay}')

    if 'txt' in format_opt:
        # Save optimized parameters to a text file
        # Get current timestamp for the filename

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")

        # Create filename with timestamp
        param_save_path = f"{output_dir}/optimized_params_{timestamp}.txt"

        with open(param_save_path, 'w') as f:
            f.write("Optimized Parameters:\n")
            f.write("-----------------\n")
            f.write(f"A:\n{cal_params.A_init}\n\n")
            f.write(f"B:\n{cal_params.B_init}\n\n") 
            f.write(f"C_aud:\n{cal_params.C_aud_init}\n\n")
            f.write(f"C_som:\n{cal_params.C_som_init}\n\n")
            f.write(f"K_aud: {cal_params.K_aud_init}\n")
            f.write(f"L_aud: {cal_params.L_aud_init}\n")
            f.write(f"K_som: {cal_params.K_som_init}\n")
            f.write(f"L_som: {cal_params.L_som_init}\n")
            f.write(f"Auditory sensor delay: {sensor_delay_aud}\n")
            f.write(f"Somatosensory sensor delay: {sensor_delay_som}\n")
            f.write(f"Actuator delay: {actuator_delay}\n")

        print(f"Optimized parameters saved to {param_save_path}")