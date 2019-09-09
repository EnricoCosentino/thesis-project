import numpy as np
from rescaleTempoAxis import rescaleTempoAxis

def MyRound(number):
        return round(number + 0.0001)

def tempogram_to_cyclicTempogram(tempogram, BPM, parameter = lambda:0):
    
    if (not hasattr(parameter, 'refTempo')):
        parameter.refTempo = 60 #BPM
    if (not hasattr(parameter, 'octave_divider')):
        parameter.octave_divider = 30
    
    if (not np.isrealobj(tempogram)):
        tempogram = np.abs(tempogram)
    
    refTempo = parameter.refTempo
    refOctave = refTempo/np.min(BPM)
    minOctave = MyRound(np.log2(np.min(BPM)/refTempo))
    maxOctave = MyRound(np.log2(np.max(BPM)/refTempo)) + 1

    #rescale to log tempo axis tempogram. Each octave is represented by
    #parameter.octave_divider tempi
    
    logBPM = refTempo * np.power(2, np.arange(minOctave, (maxOctave - 1/parameter.octave_divider) + 0.0001, 1/parameter.octave_divider))
    logAxis_tempogram, logBPM = rescaleTempoAxis(tempogram, BPM, logBPM)
    
    #cyclic projection of log axis tempogram to the reference octave
    
    def getIndices(arr, func):
        return [i for (i, val) in enumerate(arr) if func(val)]
    
    endPos = np.max(getIndices(logBPM, lambda x: x < np.max(BPM)))
    cyclicTempogram = np.zeros((parameter.octave_divider, logAxis_tempogram.shape[1]))
    for i in range(0, parameter.octave_divider):
        cyclicTempogram[i, :] = np.mean(logAxis_tempogram[i:endPos:parameter.octave_divider, :], axis = 0)
        
    cyclicAxis = refOctave * logBPM[:parameter.octave_divider + 1] / refTempo
    return cyclicTempogram, cyclicAxis