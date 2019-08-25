import numpy as np

def compute_fourierCoefficients_python(s, win, noverlap, f, fs = 1):
    
    win_len = len(win)
    