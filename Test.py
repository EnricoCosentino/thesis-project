import numpy as np

from scipy.io import wavfile
from scipy.signal import resample
from audio_to_noveltyCurve import audio_to_noveltyCurve
from noveltyCurve_to_tempogram_via_DFT import noveltyCurve_to_tempogram_via_DFT

samplerate, data = wavfile.read("02 - Moar Ghosts n' Stuff.wav")

data = (data[:,0] + data[:,1])/2
rd = resample(data, len(data)//2) #reduced data
#Da dove prendere il samplerate di rd?

parameter_novCurve = lambda:0
novCurve, featureRate = audio_to_noveltyCurve(rd, samplerate/2, parameter_novCurve)

parameter_tempogram = lambda:0
parameter_tempogram.featureRate = featureRate
parameter_tempogram.tempoWindow = 8 #in sec
parameter_tempogram.BPM = np.arange(30, 601)
#parameter_tempogram.useImplementation = 2
tempogram, T, BPM = noveltyCurve_to_tempogram_via_DFT(novCurve, parameter_tempogram) 
