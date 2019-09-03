from warnings import warn
from compute_fourierCoefficients_python import compute_fourierCoefficients_python
import sys

import numpy as np
import math
from matplotlib.pyplot import specgram

#Makes Python rounding act like MATLAB's
def MyRound(number):
        return round(number + 0.0001)

def noveltyCurve_to_tempogram_via_DFT(novelty, parameter = lambda:0):
    if (not hasattr(parameter, 'featureRate')):
        warn('parameter.featureRate not set! Assuming 1!')
        parameter.featureRate = 1
    if (not hasattr(parameter, 'tempoWindow')):
        parameter.tempoWindow = 6 #in seconds
    if (not hasattr(parameter, 'BPM')):
        parameter.BPM = np.arange(30, 601)
    if (not hasattr(parameter, 'stepsize')):
        parameter.stepsize = np.ceil(np.divide(parameter.featureRate,5)) #5 hz default
    if (not hasattr(parameter, 'useImplementation')):
        parameter.useImplementation = 2 #1: C implementation, 2: Python implementation, 3: spectrogram via goertzel algorithm
        #Only 2 currently works
    
    win_len = MyRound(parameter.tempoWindow * parameter.featureRate)
    win_len = win_len + math.fmod(win_len,2) - 1
    parameter.tempoRate = parameter.featureRate/parameter.stepsize
    
    handle = lambda n: 0.5-0.5*np.cos(2*np.pi*(np.transpose(np.arange(0, n)/(n-1))))

    windowTempogram = handle(win_len)
    
    aux_novelty = np.zeros((1,int(MyRound(win_len/2))))
    aux_novelty = np.append(aux_novelty, novelty)
    aux_novelty = np.append(aux_novelty,  np.zeros((1,int(MyRound(win_len/2)))))
    novelty = aux_novelty

    if parameter.useImplementation == 1: #C implementation
        warn("C implementation not done yet")
        sys.exit(1)
    elif parameter.useImplementation == 2: #Python implementation
        tempogram, BPM, T = compute_fourierCoefficients_python(novelty, windowTempogram, win_len - parameter.stepsize, np.divide(parameter.BPM, 60), parameter.featureRate)
        tempogram = np.divide(tempogram, np.sqrt(win_len))/ sum(windowTempogram) * win_len
    elif parameter.useImplementation == 3:
        warn("This implementation is still being worked on and does not work as intended")
        BPM, T, tempogram = specgram(novelty, Fs = parameter.featureRate, window = windowTempogram, noverlap = win_len - parameter.stepsize, mode = 'psd')
        tempogram = np.divide(tempogram, np.sqrt(win_len))/ sum(windowTempogram) * win_len

    BPM = np.multiply(BPM, 60)
    T = T - T[0]
    
    return tempogram, T, BPM
    