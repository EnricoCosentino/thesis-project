import numpy as np

from scipy.io import wavfile
from scipy.signal import resample
from audio_to_noveltyCurve import audio_to_noveltyCurve
from noveltyCurve_to_tempogram_via_DFT import noveltyCurve_to_tempogram_via_DFT
from normalizeFeature import normalizeFeature
from tempogram_to_PLPcurve import tempogram_to_PLPcurve
from noveltyCurve_to_tempogram_via_ACF import noveltyCurve_to_tempogram_via_ACF
from rescaleTempoAxis import rescaleTempoAxis
from tempogram_to_cyclicTempogram import tempogram_to_cyclicTempogram

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
parameterTempogram = lambda:0
parameterTempogram.featureRate = featureRate
parameterTempogram.tempoWindow = 8 #in sec
parameterTempogram.BPM = np.arange(30, 601)
#parameterTempogram.useImplementation = 2
tempogram, T, BPM = noveltyCurve_to_tempogram_via_DFT(novCurve, parameterTempogram) 
tempogram = normalizeFeature(tempogram, 2, 0.0001)

#################
#PLP CURVE
#################
parameter_PLP = lambda:0
parameter_PLP.featureRate = featureRate
parameter_PLP.tempoWindow = parameterTempogram.tempoWindow

#I valori di PLP divergono sempre di piu dalla versione MATLAB man mano che ci
#si avvicina alla fine dell'array
#Il motivo sembra essere che la posizione del massimo di alcune colonne del 
#tempogramma differisce tra le 2 versioni
PLP, featureRate = tempogram_to_PLPcurve(tempogram, T, BPM, parameter_PLP)
PLP = PLP[:novCurve.shape[1]] #removes zero padding

#################
#TEMPOGRAM VIA ACF
#################
parameterTempogram = lambda:0
parameterTempogram.featureRate = featureRate
parameterTempogram.tempoWindow = 8
parameterTempogram.maxLag = 2
parameterTempogram.minLag = 0.1
tempogram_autocorrelation_timeLag, T, timeLag = noveltyCurve_to_tempogram_via_ACF(novCurve, parameterTempogram)
tempogram_autocorrelation_timeLag = np.real(normalizeFeature(tempogram_autocorrelation_timeLag, 2, 0.0001))

#################
#RESCALED TEMPOGRAM VIA DFT
#################
tempogram_DFT_timelag, timeLag = rescaleTempoAxis(tempogram, np.divide(60, BPM), timeLag)
tempogram_DFT_timelag = normalizeFeature(tempogram_DFT_timelag, 2, 0.0001)

#################
#CYCLIC TEMPOGRAM FOURIER, 120 DIM
#################
octave_divider = 120

parameterTempogram = lambda:0
parameterTempogram.featureRate = featureRate
parameterTempogram.tempoWindow = 5
parameterTempogram.BPM = 30 *  np.power(2, np.arange(0, 4 + 0.0001, 1/octave_divider))

parameterCyclic = lambda:0
parameterCyclic.octave_divider = octave_divider
tempogram_fourier, T, BPM  = noveltyCurve_to_tempogram_via_DFT(novCurve, parameterTempogram)
cyclicTempogram_fourier, cyclicAxis = tempogram_to_cyclicTempogram(tempogram_fourier, BPM, parameterCyclic)