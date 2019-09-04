import numpy as np

from scipy.io import wavfile
from scipy.signal import resample
from audio_to_noveltyCurve import audio_to_noveltyCurve
from noveltyCurve_to_tempogram_via_DFT import noveltyCurve_to_tempogram_via_DFT
from normalizeFeature import normalizeFeature
from tempogram_to_PLPcurve import tempogram_to_PLPcurve
from noveltyCurve_to_tempogram_via_ACF import noveltyCurve_to_tempogram_via_ACF
from rescaleTempoAxis import rescaleTempoAxis

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
#TEMPOGRAM VIA DFT
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
#Il motivo sembra essere che la posizione del massimo di alcune colonne del 
#tempogramma differisce tra le 2 versioni
PLP, featureRate = tempogram_to_PLPcurve(tempogram, T, BPM, parameter_PLP)
PLP = PLP[:novCurve.shape[1]]

#################
#TEMPOGRAM VIA ACF
#################
parameter_tempogram = lambda:0
parameter_tempogram.featureRate = featureRate
parameter_tempogram.tempoWindow = 8
parameter_tempogram.maxLag = 2
parameter_tempogram.minLag = 0.1
tempogram_autocorrelation_timeLag, T, timeLag = noveltyCurve_to_tempogram_via_ACF(novCurve, parameter_tempogram)
tempogram_autocorrelation_timeLag = np.real(normalizeFeature(tempogram_autocorrelation_timeLag, 2, 0.0001))

#################
#RESCALED TEMPOGRAM VIA DFT
#################
tempogram_DFT_timelag, timeLag = rescaleTempoAxis(tempogram, np.divide(60, BPM), timeLag)
tempogram_DFT_timelag = normalizeFeature(tempogram_DFT_timelag, 2, 0.0001)
