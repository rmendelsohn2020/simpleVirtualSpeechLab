import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def data_prep(filepath, timeseries, save_path, convert_opt='multiplier2cents'):
    #Data Prep
    incoming_data = np.loadtxt(filepath, delimiter=',', skiprows=1)

    if convert_opt == 'multiplier2cents':
        # Extract columns
        time = incoming_data[:, 0]  # First column: time
        distortion_signal_multiplier = incoming_data[:, 1]  # Distortion signal in multiplier
        participant_traces_multiplier = incoming_data[:, 3:21]  # Participant traces in multiplier

        #Convert time to simulation timeseries
        time_conv = np.zeros(len(time))
        for i in range(len(time)):
            time_conv[i] = 2.06+time[i]

        # Convert to cents
        distortion_signal_cents = 1200 * np.log2(distortion_signal_multiplier)
        participant_traces_cents = 1200 * np.log2(participant_traces_multiplier)
        
        #interpolate to timeseries
        time_conv_interp,distortion_signal_cents_interp = data_interp(np.array([time_conv, distortion_signal_cents]), timeseries, trace_column=1, showplt=False)
        participant_traces_cents_interp = np.zeros((len(timeseries), participant_traces_cents.shape[1]))
        for i in range(participant_traces_cents.shape[1]):
            participant_traces_cents_interp[:, i] = data_interp(np.array([time_conv, participant_traces_cents[:,i]]), timeseries, trace_column=1, showplt=False)[1]
        # #Plot 1st participant's data and distortion signal
        # plt.figure(figsize=(10, 6))
        # plt.plot(time, distortion_signal_cents, label='Distortion Signal')
        # first_participant = participant_traces_cents[:, 0]
        # plt.plot(time, first_participant, label='Participant 1')

        # Create .csv file with cents data
        data_cents = np.column_stack((time_conv, distortion_signal_cents, participant_traces_cents))
        data_path= os.path.join(save_path,'data_cents.csv')

        # Create directory if it doesn't exist
        data_dir = os.path.dirname(data_path)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Save the data to CSV
        np.savetxt(data_path, data_cents, delimiter=',', 
                   header='Time,Distortion Signal (cents),Participant 1 (cents),Participant 2 (cents),Participant 3 (cents),Participant 4 (cents),Participant 5 (cents),Participant 6 (cents),Participant 7 (cents),Participant 8 (cents),Participant 9 (cents),Participant 10 (cents),Participant 11 (cents),Participant 12 (cents),Participant 13 (cents),Participant 14 (cents),Participant 15 (cents),Participant 16 (cents),Participant 17 (cents),Participant 18 (cents),Participant 19 (cents)', 
                   comments='')

        #Create .csv file with interpolated cents data
        data_cents_interp = np.column_stack((time_conv_interp, distortion_signal_cents_interp, participant_traces_cents_interp))
        data_path_interp= os.path.join(save_path,'data_cents_interp.csv')
        np.savetxt(data_path_interp, data_cents_interp, delimiter=',', header='Time,Distortion Signal (cents),Participant 1 (cents),Participant 2 (cents),Participant 3 (cents),Participant 4 (cents),Participant 5 (cents),Participant 6 (cents),Participant 7 (cents),Participant 8 (cents),Participant 9 (cents),Participant 10 (cents),Participant 11 (cents),Participant 12 (cents),Participant 13 (cents),Participant 14 (cents),Participant 15 (cents),Participant 16 (cents),Participant 17 (cents),Participant 18 (cents),Participant 19 (cents)', comments='')
        plot_calibration_data(save_path, 'data_cents_interp.csv')
    return data_path_interp
    # Load the .csv file with cents data and interpolate the points
    #data_converted = np.loadtxt('data_cents.csv', delimiter=',', skiprows=1)
    #data_interpolated = data_interp(data_cents, timeseries, trace_column=1, showplt=True)
        

def data_interp(filepath_or_vals, timeseries, trace_column=2, showplt=True, save_path=None):
    #datDir = '/Users/rachelmendelsohn/Desktop/Salk/ArticulatoryFeedback/Figures/Simulations/2024_07_10_1139_39'
    

    # Separate the data into x and y values
    if isinstance(filepath_or_vals, str):
        # Step 1: Read the file
        # Assuming the  file has two columns: x and y values separated by commas
        data = np.loadtxt(filepath_or_vals, delimiter=',', skiprows=1)
        x = data[:, 0]
        y = data[:, trace_column]
    elif isinstance(filepath_or_vals, np.ndarray):
        x = filepath_or_vals[0]
        y = filepath_or_vals[trace_column]

    # print('x shape:', x.shape)
    # print('y shape:', y.shape)
    # print('timeseries shape:', timeseries.shape)
    #Plot the original points
    # plt.figure(figsize=(10, 6))
    # plt.plot(x, y, 'o', label='Original Points')
    

    # Step 2: Interpolate the points
    # Create an interpolation function
    def custom_interp(x_new, x, y):
        # Create an interpolation function for the valid range
        interp_func = interp1d(x, y, kind='quadratic', fill_value='extrapolate')
        
        # Interpolate the new y values
        y_new = interp_func(x_new)
        
        # Apply the first y value to any x_new values before the minimum x value
        y_new[x_new < x.min()] = y[0]
        
        return y_new

    # Define new x values for interpolation
    x_new = timeseries
    y_new = custom_interp(x_new, x, y)

    if showplt:
        # Step 3: Plot the original points and the interpolated curve
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'o', label='Original Points')
        plt.plot(x_new, y_new, '-', label='Interpolated Curve')
        plt.xlabel('X values')
        plt.ylabel('Y values')
        plt.title('Interpolation of Data Points')
        plt.legend()
        plt.grid(True)
        plt.show()

    return x_new, y_new

def plot_calibration_data(filepath, filename):
    # Load the CSV file using numpy
    fullpath = filepath + '/' + filename
    data = np.loadtxt(fullpath, delimiter=',', skiprows=1)
    
    # Create a new figure
    plt.figure(figsize=(11, 7))

    # Extract columns
    time = data[:, 0]  # First column: time
    distortion_signal_multiplier = data[:, 1]  # Distortion signal 
    participant_traces_multiplier = data[:, 2:19]  # Participant traces 


    # Create the plot
    plt.plot(time, distortion_signal_multiplier, label='Distortion Signal', color='black', linewidth=2)
    for i in range(participant_traces_multiplier.shape[1]):
        plt.plot(time, participant_traces_multiplier[:, i], label=f'Participant {i+1}', linewidth=1)

    # Add labels and title
    plt.xlabel('Time (s)')
    plt.ylabel('Cents')
    plt.title(filename)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    # Adjust layout and display the plots
    #plt.tight_layout()
    

    save_path = filepath
    save_name = filename.replace('.csv', '') + '.png'
    fullsavepath = save_path + '/' + save_name
    print('fullsavepath:', fullsavepath)
    plt.savefig(fullsavepath)

    plt.show()

def truncate_data(x_data, y_data, t_start, t_end):
    # Truncate the data to the specified time range
    x_data_truncated = x_data[(x_data >= t_start) & (x_data <= t_end)]
    print('pre-truncation x:', x_data.shape)
    print('post-truncation x:', x_data_truncated.shape)

    if y_data is not None:
        y_data_truncated = y_data[(x_data >= t_start) & (x_data <= t_end)]
        # #Plot the truncated data
        # plt.figure(figsize=(12, 6))
        # plt.plot(x_data_truncated, y_data_truncated, label='Truncated Data')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Cents')
        # plt.title('Truncated Data')
    else:
        y_data_truncated = None
        print('truncated timeseries:', x_data_truncated)
    

    return x_data_truncated, y_data_truncated