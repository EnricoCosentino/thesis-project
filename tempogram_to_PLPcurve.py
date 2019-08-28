import numpy as np
import sys

from warnings import warn

def MyRound(number):
        return round(number + 0.0001)

def tempogram_to_PLPcurve(tempogram, T, BPM, parameter = lambda:0):
    if (not hasattr(parameter,'featureRate')):
        warn('featureRate unset! Assuming 100!')
        parameter.featureRate = 100
    if (not hasattr(parameter,'tempoWindow')):
        warn('tempo window length unset! Assuming 6 sec.')
        parameter.tempoWindow = 6 #sec
    if (not hasattr(parameter,'useTempocurve')):
        parameter.useTempocurve = 0
    if (not hasattr(parameter,'tempocurve')):
        parameter.tempocurve = 0
    if (not hasattr(parameter,'PLPrange')):
        parameter.PLPrange = [BPM[0], BPM[-1]]
    if (not hasattr(parameter,'stepsize')):
        parameter.stepsize = np.ceil(np.divide(parameter.featureRate,5)) # 5 Hz default
    
    handle = lambda n: 0.5-0.5*np.cos(2*np.pi*(np.transpose(np.arange(0, n))/(n-1)))
    
    if np.isrealobj(tempogram):
        #Come inviare messaggi di errore su Python?
        print('Complex valued fourier tempogram needed for computing PLP curves!')
        sys.exit(1)
        
    tempogramAbs = np.abs(tempogram)
    rangeAr = np.zeros(2, dtype = 'int')
    rangeAr[0] = np.argmin(np.abs(BPM-parameter.PLPrange[0]))
    rangeAr[1] = np.argmin(np.abs(BPM-parameter.PLPrange[1]))

    local_max = np.zeros(tempogramAbs.shape[1])
    if (not parameter.useTempocurve):
        for frame in np.arange(0, tempogramAbs.shape[1], dtype = 'int'):
            local_max[frame] = np.argmax(tempogramAbs[rangeAr[0]:rangeAr[1], frame])
            local_max[frame] = local_max[frame] + rangeAr[0]
    else:
        for frame in np.arange(0, tempogramAbs.shape[1], dtype = 'int'):
            idx = np.argmin(np.abs(BPM - parameter.tempocurve[frame]))
            local_max[frame] = idx
            
    win_len = MyRound(np.multiply(parameter.tempoWindow, parameter.featureRate))
    win_len = win_len + np.mod(win_len,2) - 1
    
    t = np.multiply(T, parameter.featureRate)
    # if novelty curve is zero padded with half a window on both sides, PLP is
    # always larger than novelty
    PLP = np.zeros(tempogram.shape[1]*int(parameter.stepsize))
    
    window = handle(win_len)
    # normalize window so sum(window) == len(window), like it is for box window
    window = np.divide(window, sum(window)/win_len)

    #normalize window accordign to overlap, this guarantees that max(PLP)<=1
    window = np.divide(window, win_len/parameter.stepsize)

    ##################
    #OVERLAP ADD
    ##################
    
    for frame in np.arange(0, tempogram.shape[1]):
        
        t0 = int(np.ceil(t[frame] - win_len/2))
        t1 = int(np.floor(t[frame] + win_len/2))

        phase = np.angle(tempogram[int(local_max[frame]), frame])
        Tperiod = parameter.featureRate*60/BPM[int(local_max[frame])] #period length
        length = (t1 - t0 + 1)/Tperiod #How many periods?
        
        aux_arr = np.arange(0, length-1/Tperiod, 1/Tperiod)
        if (not aux_arr.shape[:] == window.shape[:]):
           aux_arr = np.append(aux_arr, length-1/Tperiod)
        #Unfortunately it allocates the whole array twice
        
        cosine = np.multiply(window, np.cos(aux_arr*2*np.pi + phase))

        if t0 < 1:
            cosine = cosine[-t0+1:]
            t0 = 1
        
        if t1 > PLP.shape[0]:
            cosine = cosine[:-(t1-PLP.shape[0])]
            aux = t1
            t1 = PLP.shape[0]
            
        PLP[t0-1:t1] = PLP[t0-1:t1] + cosine 

    PLP[PLP < 0] = 0
    featureRate = parameter.featureRate
    return PLP, featureRate