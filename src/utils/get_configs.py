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
            data_path = config_paths['paths']['calibration_data']['mean_aud_filename']
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

    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)

    if not os.path.exists(data_path):
        unprocessed_data_path = config_paths['paths']['calibration_data']['unprocessed_indiv_files'][pert_type]['path']
        add_time_column_to_csv(unprocessed_data_path, filename, data_path)

    return type('PathObject', (), {'data_save_path': data_save_path, 'data_path': data_path, 'fig_save_path': fig_save_path})

def get_params():
     # Initialize the config loader
    with open("src/configs/paths.yaml", "r") as f:
        config_paths = yaml.safe_load(f)
    with open("src/configs/experiment.yaml", "r") as f:
        config_expt = yaml.safe_load(f)

    # Define simulation parameters
    trace_type = config_expt['data']['trace_type']
    participant_ID = config_expt['data']['participant_ID']

    duration = config_expt['simulation']['duration']
    dt = config_expt['simulation']['sec_per_step']
    ref_type = config_expt['simulation']['ref_type']

    pert_type = config_expt['perturbation']['type']
    pert_mag = config_expt['perturbation']['magnitude']
    pert_onset = config_expt['perturbation']['onset']
    pert_duration = config_expt['perturbation']['duration']
    ramp_up_duration = config_expt['perturbation']['ramp']['up_duration']
    ramp_down_duration = config_expt['perturbation']['ramp']['down_duration']

    actuator_delay = config_expt['starting_params']['delays']['actuator']
    sensor_delay_aud = config_expt['starting_params']['delays']['sensor']['auditory']
    sensor_delay_som = config_expt['starting_params']['delays']['sensor']['somatosensory']

    A_init = np.array(config_expt['starting_params']['system']['A'])
    B_init = np.array(config_expt['starting_params']['system']['B'])
    C_init = np.array(config_expt['starting_params']['system']['C'])
    R_init = config_expt['starting_params']['system']['R_val']
    RN_init = config_expt['starting_params']['system']['RN_val']

    return type('ParamsObject', (), {'trace_type': trace_type, 'participant_ID': participant_ID, 'duration': duration, 'dt': dt, 'ref_type': ref_type, 'pert_type': pert_type, 'pert_mag': pert_mag, 'pert_onset': pert_onset, 'pert_duration': pert_duration, 'ramp_up_duration': ramp_up_duration, 'ramp_down_duration': ramp_down_duration, 'actuator_delay': actuator_delay, 'sensor_delay_aud': sensor_delay_aud, 'sensor_delay_som': sensor_delay_som, 'A_init': A_init, 'B_init': B_init, 'C_init': C_init, 'R_init': R_init, 'RN_init': RN_init})

