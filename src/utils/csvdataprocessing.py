import pandas as pd
import numpy as np
import os
import yaml


def add_time_column_to_csv(file_path, file_name, save_path, dt=0.005, save_as_new=True):
    # Load the CSV
    df = pd.read_csv(file_path + '/' + file_name)

    # Identify index of first non-zero value in the perturbation column
    perturbation = df.iloc[:, 0]
    nonzero_index = perturbation.ne(0).idxmax()

    # Compute time array
    num_rows = len(df)
    times = (np.arange(num_rows) - (nonzero_index + 1)) * dt

    # Insert time column at the beginning
    df.insert(0, 'time', times)

    # Save to file
    if save_as_new:
        new_file_path = save_path + 'secs_' + file_name 
    else:
        new_file_path = save_path + file_name

    df.to_csv(new_file_path, index=False)
    print(f"Updated file saved to: {new_file_path}")

with open("src/configs/paths.yaml", "r") as f:
    config_paths = yaml.safe_load(f)

# Example usage on one file
save_path = config_paths['paths']['save_path'] + '/AudPert/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

filename = config_paths['paths']['calibration_data']['unprocessed_indiv_files']['auditory']['basename'] +str(1)+ '.csv'
filepath = config_paths['paths']['calibration_data']['unprocessed_indiv_files']['auditory']['path']
add_time_column_to_csv(filepath, filename, save_path)

# Or batch process all files in a folder
# for filename in os.listdir("data_folder"):
#     if filename.endswith(".csv"):
#         add_time_column_to_csv(os.path.join("data_folder", filename))
