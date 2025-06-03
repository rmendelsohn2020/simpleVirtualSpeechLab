import pandas as pd
import numpy as np

def get_perturbation_event_times(file_path, epsilon=1e-10):
    df = pd.read_csv(file_path)

    time_col=0
    perturb_col=1

    # Get the series
    time = df.iloc[:, time_col]
    perturbation = df.iloc[:, perturb_col]

    # Find index of first non-zero perturbation using epsilon tolerance
    first_nonzero_idx = (perturbation[abs(perturbation) > epsilon].index[0]) - 1
    print('first_nonzero_idx', first_nonzero_idx)

    # Find index of maximum absolute value (peak of perturbation)
    max_abs_val = abs(perturbation).max()
    print('max_abs_val', max_abs_val)
    # Find the actual value (could be negative) at the maximum absolute point
    max_idx = (perturbation[abs(perturbation) > (max_abs_val - epsilon)].index[0])
    print('max_idx', max_idx)
    # Get times
    time_at_first_nonzero = time.iloc[first_nonzero_idx]
    time_at_maximum = time.iloc[max_idx]

    return time_at_first_nonzero, time_at_maximum
