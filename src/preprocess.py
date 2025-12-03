
import numpy as np

def preprocess_input(data_dict):
    return np.array(list(data_dict.values()), dtype=float)
