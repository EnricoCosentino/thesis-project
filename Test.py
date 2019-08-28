import numpy as np

from scipy.io import wavfile
from scipy.signal import resample
from audio_to_noveltyCurve import audio_to_noveltyCurve
from noveltyCurve_to_tempogram_via_DFT import noveltyCurve_to_tempogram_via_DFT
from normalizeFeature import normalizeFeature
from tempogram_to_PLPcurve import tempogram_to_PLPcurve

samplerate, data = wavfile.read("02 - Moar Ghosts n' Stuff.wav")

data = (data[:,0] + data[:,1])/2
rd = resample(data, len(data)//2) #reduced data
#Da dove prendere il samplerate di rd?


#################
#NOVELTY CURVE
#################
parameter_novCurve = lambda:0
novCurve, featureRate = audio_to_noveltyCurve(rd, samplerate/2, parameter_novCurve)

#################
#TEMPOGRAM
#################
parameter_tempogram = lambda:0
parameter_tempogram.featureRate = featureRate
parameter_tempogram.tempoWindow = 8 #in sec
parameter_tempogram.BPM = np.arange(30, 601)
#parameter_tempogram.useImplementation = 2
tempogram, T, BPM = noveltyCurve_to_tempogram_via_DFT(novCurve, parameter_tempogram) 
tempogram = normalizeFeature(tempogram, 2, 0.0001)

#################
#PLP CURVE
#################
parameter_PLP = lambda:0
parameter_PLP.featureRate = featureRate
parameter_PLP.tempoWindow = parameter_tempogram.tempoWindow

#I valori di PLP divergono sempre di piu dalla versione MATLAB man mano che ci
#si avvicina alla fine dell'array
PLP, featureRate = tempogram_to_PLPcurve(tempogram, T, BPM, parameter_PLP)
