import numpy as np

from scipy.io import wavfile
from scipy.signal import resample
from audio_to_noveltyCurve import audio_to_noveltyCurve
from noveltyCurve_to_tempogram_via_DFT import noveltyCurve_to_tempogram_via_DFT
from tempogram_to_PLPcurve import tempogram_to_PLPcurve
from sonify_noveltyCurve import sonify_noveltyCurve

samplerate, data = wavfile.read("02 - Moar Ghosts n' Stuff.wav")

data = (data[:,0] + data[:,1])/2
rd = resample(data, len(data)//2) #reduced data

#################
#NOVELTY CURVE
#################
parameter_novCurve = lambda:0
novCurve, featureRate = audio_to_noveltyCurve(rd, samplerate/2, parameter_novCurve)

#################
#TEMPOGRAM VIA DFT
#################
parameterTempogram = lambda:0
parameterTempogram.featureRate = featureRate
parameterTempogram.tempoWindow = 6 #in sec
parameterTempogram.BPM = np.arange(30, 601)

tempogram, T, BPM = noveltyCurve_to_tempogram_via_DFT(novCurve, parameterTempogram) 

#################
#PLP CURVE
#################
parameter_PLP = lambda:0
parameter_PLP.featureRate = featureRate
parameter_PLP.tempoWindow = parameterTempogram.tempoWindow

PLP, featureRate = tempogram_to_PLPcurve(tempogram, T, BPM, parameter_PLP)
PLP = PLP[:novCurve.shape[1]] #removes zero padding

#################
#SONIFY
#################
parameterSoni = lambda:0
parameterSoni.Fs = samplerate/2
parameterSoni.featureRate = featureRate

peaksl, sonification = sonify_noveltyCurve(novCurve, rd, parameterSoni)
sonification = sonification.T

wavfile.write("sonification_novelty.wav", int(samplerate/2), sonification)
