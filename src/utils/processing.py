import numpy as np

def set_array_value(array, index, value):
    array[index] = value
    return array

def round_to_timesteps(value):
    return int(value)

def secs_to_timesteps(value, dt):
    return int(value/dt)

def make_jsonable_dict(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            if v.size == 1:
                out[k] = float(v)
            else:
                out[k] = v.tolist()
        else:
            out[k] = v
    return out

