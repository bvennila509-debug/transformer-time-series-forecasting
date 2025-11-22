"""utils.py
Helper functions for sliding windows, metrics, and data preparation.
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def rmse(a,b):
    return mean_squared_error(a,b, squared=False)

def mae(a,b):
    return mean_absolute_error(a,b)

def sliding_windows(data, seq_len, pred_len):
    """Yield (input_seq, target_seq) windows from a 2D numpy array [T, features].
    input_seq shape: (seq_len, features), target_seq shape: (pred_len, target_dim)
    """
    T = data.shape[0]
    for start in range(0, T - seq_len - pred_len + 1):
        x = data[start:start+seq_len]
        y = data[start+seq_len:start+seq_len+pred_len, 0]  # assume column 0 is target
        yield x, y
