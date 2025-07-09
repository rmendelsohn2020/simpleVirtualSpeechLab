import yaml
import os
import numpy as np
from utils.pitchpert_dataprep import add_time_column_to_csv

def get_paths():
    # Initialize the config loader
    with open("src/configs/paths.yaml", "r") as f:
        config_paths = yaml.safe_load(f)
    with open("src/configs/experiment.yaml", "r") as f:
        config_expt = yaml.safe_load(f)

    # Define data paths and save paths
    fig_save_path = config_paths['paths']['fig_save_path']
    

    if config_expt['data']['trace_type'] == 'mean':
        if config_expt['perturbation']['type'] == 'auditory':
            data_save_path = config_paths['paths']['data_save_path'] + '/AudPert/'
            data_dir = config_paths['paths']['calibration_data']['directory']
            filename = config_paths['paths']['calibration_data']['mean_aud_filename']
            data_path = os.path.join(data_dir, filename)
        else:
            print('Mean trace data only available for auditory perturbations')


    elif config_expt['data']['trace_type'] == 'single trial':
        pert_type = config_expt['perturbation']['type']
        if pert_type == 'auditory':
            data_save_path = config_paths['paths']['data_save_path'] + '/AudPert/'
            data_dir = config_paths['paths']['calibration_data']['processed_indiv_files'][pert_type]['path']
            indiv = config_expt['data']['participant_ID']
            filename = config_paths['paths']['calibration_data']['processed_indiv_files'][pert_type]['basename'] + str(indiv) + '.csv'
            data_path = os.path.join(data_dir, filename)

        elif 'laryngeal' in pert_type:
            data_save_path = config_paths['paths']['data_save_path'] + '/LarPert/'
            data_dir = config_paths['paths']['calibration_data']['processed_indiv_files'][pert_type]['path']
            indiv = config_expt['data']['participant_ID']
            filename = config_paths['paths']['calibration_data']['processed_indiv_files'][pert_type]['basename'] + str(indiv) + '.csv'
            data_path = os.path.join(data_dir, filename)

        elif 'state' in pert_type:
            data_save_path = config_paths['paths']['data_save_path'] + '/StatePert/'
            print('State pert, no available data')
            
        else:
            raise ValueError(f"Invalid perturbation type: {pert_type}")


    else:
        raise ValueError(f"Invalid trace type: {config_expt['data']['trace_type']}")
    
    if config_paths['paths']['interp_folder'] in data_path:
        interpolated = True
    else:
        interpolated = False

    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)

    if not os.path.exists(data_path):
        unprocessed_data_path = config_paths['paths']['calibration_data']['unprocessed_indiv_files'][pert_type]['path']
        add_time_column_to_csv(unprocessed_data_path, filename, data_path)

    return type('PathObject', (), {'data_save_path': data_save_path, 'data_path': data_path, 'fig_save_path': fig_save_path, 'interpolated': interpolated})

def get_params():
     # Initialize the config loader
    with open("src/configs/paths.yaml", "r") as f:
        config_paths = yaml.safe_load(f)
    with open("src/configs/experiment.yaml", "r") as f:
        config_expt = yaml.safe_load(f)

    # Define simulation parameters
    trace_type = config_expt['data']['trace_type']
    participant_ID = config_expt['data']['participant_ID']
    if config_expt['data']['trace_type'] == 'mean':
        trial_ID = participant_ID
    else:
        trial_ID = config_expt['data']['trial_ID']  

    duration = config_expt['simulation']['duration']
    dt = config_expt['simulation']['sec_per_step']
    ref_type = config_expt['simulation']['ref_type']

    pert_type = config_expt['perturbation']['type']
    pert_mag = config_expt['perturbation']['magnitude']
    pert_onset = config_expt['perturbation']['onset']
    pert_duration = config_expt['perturbation']['duration']
    ramp_up_duration = config_expt['perturbation']['ramp']['up_duration']
    ramp_down_duration = config_expt['perturbation']['ramp']['down_duration']

    system_type = config_expt['simulation']['system_type']

    if system_type == 'Template':
        actuator_delay = config_expt['starting_params']['delays']['actuator']
        sensor_delay_aud = config_expt['starting_params']['delays']['sensor']['auditory']
        sensor_delay_som = config_expt['starting_params']['delays']['sensor']['somatosensory']

        A_init = np.array(config_expt['starting_params']['system']['A'])
        B_init = np.array(config_expt['starting_params']['system']['B'])
        C_aud_init = np.array(config_expt['starting_params']['system']['C_aud'])
        C_som_init = np.array(config_expt['starting_params']['system']['C_som'])
        K_aud_init = config_expt['starting_params']['system']['K_aud']
        L_aud_init = config_expt['starting_params']['system']['L_aud']
        Kf_aud_init = config_expt['starting_params']['system']['Kf_aud']
        K_som_init = config_expt['starting_params']['system']['K_som']
        L_som_init = config_expt['starting_params']['system']['L_som']
        Kf_som_init = config_expt['starting_params']['system']['Kf_som']

        # tune_Rs = config_expt['starting_params']['system']['tune_Rs']
        # tune_RNs = config_expt['starting_params']['system']['tune_RNs']

        return type('ParamsObject', (), {'trace_type': trace_type, 'participant_ID': participant_ID, 'trial_ID': trial_ID, 'duration': duration, 'dt': dt, 'system_type': system_type, 'ref_type': ref_type, 'pert_type': pert_type, 'pert_mag': pert_mag, 'pert_onset': pert_onset, 'pert_duration': pert_duration, 'ramp_up_duration': ramp_up_duration, 'ramp_down_duration': ramp_down_duration, 'actuator_delay': actuator_delay, 'sensor_delay_aud': sensor_delay_aud, 'sensor_delay_som': sensor_delay_som, 'A_init': A_init, 'B_init': B_init, 'C_aud_init': C_aud_init, 'C_som_init': C_som_init, 'K_aud_init': K_aud_init, 'L_aud_init': L_aud_init, 'Kf_aud_init': Kf_aud_init, 'K_som_init': K_som_init, 'L_som_init': L_som_init, 'Kf_som_init': Kf_som_init})
    elif system_type == 'DIVA':
        kearney_name = config_expt['diva_starting_params']['kearney_name']
        tau_A_init = config_expt['diva_starting_params']['delays']['tau_A'] #D1-D15
        tau_S_init = config_expt['diva_starting_params']['delays']['tau_S'] #D2,
        tau_As_init = config_expt['diva_starting_params']['delays']['tau_As']
        tau_Ss_init = config_expt['diva_starting_params']['delays']['tau_Ss']

        alpha_A_init = config_expt['diva_starting_params']['gains']['alpha_A'] #D1-D15
        alpha_S_init = config_expt['diva_starting_params']['gains']['alpha_S'] #D1,D2,D5-D10,D12-D15
        alpha_As_init = config_expt['diva_starting_params']['gains']['alpha_As']
        alpha_Ss_init = config_expt['diva_starting_params']['gains']['alpha_Ss']
        alpha_Av_init = config_expt['diva_starting_params']['gains']['alpha_Av']
        alpha_Sv_init = config_expt['diva_starting_params']['gains']['alpha_Sv']

        return type('DivaParamsObject', (), {'trace_type': trace_type, 'participant_ID': participant_ID, 'trial_ID': trial_ID, 'duration': duration, 'dt': dt, 'system_type': system_type, 'ref_type': ref_type, 'pert_type': pert_type, 'pert_mag': pert_mag, 'pert_onset': pert_onset, 'pert_duration': pert_duration, 'ramp_up_duration': ramp_up_duration, 'ramp_down_duration': ramp_down_duration, 'kearney_name': kearney_name, 'tau_A_init': tau_A_init, 'tau_S_init': tau_S_init, 'tau_As_init': tau_As_init, 'tau_Ss_init': tau_Ss_init, 'alpha_A_init': alpha_A_init, 'alpha_S_init': alpha_S_init, 'alpha_As_init': alpha_As_init, 'alpha_Ss_init': alpha_Ss_init, 'alpha_Av_init': alpha_Av_init, 'alpha_Sv_init': alpha_Sv_init})