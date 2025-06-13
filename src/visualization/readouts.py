from utils.get_configs import get_paths
from datetime import datetime

def readout_optimized_params(cal_params, sensor_delay_aud, sensor_delay_som, actuator_delay, format_opt=['txt','print']):
    path_obj = get_paths()
    fig_save_path = path_obj.fig_save_path

    if 'print' in format_opt:
        print('Optimized parameters:')
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
        param_save_path = f"{fig_save_path}/optimized_params_{timestamp}.txt"



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