from scipy import signal

import numpy as np
import numpy.matlib
import resampy

from audio_to_spectrogram_via_STFT import audio_to_spectrogram_via_STFT

def SetMaxIn2DArray(arr, thresh):
    for (i,j), val in np.ndenumerate(arr):
        arr[i][j] = max(val, thresh)
    
    return arr

def SetMaxIn1DArray(arr, thresh):
    for i, val in np.ndenumerate(arr):
        arr[i] = max(val, thresh)
    
    return arr

def SetMinIn1DArray(arr, thresh):
    for i, val in np.ndenumerate(arr):
        arr[i] = min(val, thresh)
    
    return arr

#Makes Python rounding act like MATLAB's
def MyRound(number):
        return round(number + 0.0001)

#Thanks to user jowlo on StackOverflow
def conv2(x, y, mode='same'):
    return np.rot90(signal.convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)

def resample_noveltyCurve(noveltyCurve, parameter):
    p = MyRound(1000*parameter.resampleFeatureRate/parameter.featureRate)
    noveltyCurve = resampy.resample(noveltyCurve, 1000, p)
    featureRate = parameter.featureRate*p/1000
    return noveltyCurve, featureRate
    
def novelty_smoothedSubtraction(noveltyCurve, parameter):
    handle = lambda n: 0.5-0.5*np.cos(2*np.pi*(np.transpose(np.arange(0, n)/(n-1))))
    smooth_len = 1.5
    smooth_len = max(np.ceil(smooth_len*parameter.fs/parameter.stepsize), 3)
    smooth_filter = handle(smooth_len)
    local_average = conv2(noveltyCurve.reshape((1,noveltyCurve.shape[0])), np.rot90(np.divide(smooth_filter, sum(smooth_filter)).reshape((smooth_filter.shape[0],1))))
    
    noveltySub = noveltyCurve - local_average
    noveltySub[noveltySub < 0] = 0
    return noveltySub, local_average


def audio_to_noveltyCurve(f_audio, fs, parameter=lambda:0):

    handle = lambda n: 0.5-0.5*np.cos(2*np.pi*(np.transpose(np.arange(0, n)/(n-1))))
    
    #Default inizialization of parameter
    parameter.fs = fs
    if (not hasattr(parameter, 'win_len')) or parameter.win_len == 0:
        parameter.win_len = 1024*fs/22050
    if (not hasattr(parameter, 'stepsize')) or parameter.stepsize == 0:
        parameter.stepsize = 512*fs/22050
    if (not hasattr(parameter,'compressionC')) or parameter.compressionC == 0:
        parameter.compressionC = 1000
    if (not hasattr(parameter,'logCompression')) or parameter.logCompression == 0:
        parameter.logCompression = True
    if (not hasattr(parameter,'resampleFeatureRate')) or parameter.resampleFeatureRate == 0:
        parameter.resampleFeatureRate = 200

    parameter.returnMagSpec = 1;
    parameter.StftWindow = handle(parameter.win_len)
    specData, featureRate = audio_to_spectrogram_via_STFT(f_audio, parameter)
    parameter.featureRate = featureRate
    
    specData = np.divide(specData, np.max(np.max(specData)))
    thresh = -74 # dB
    thresh = 10**(thresh/20) #Perche nell'originale il prodotto e puntuale?
    specData = SetMaxIn2DArray(specData, thresh)
    
    bands = np.array([[0,500], [500,1250], [1250,3125], [3125,7812.5], [7812.5, np.floor(np.divide(parameter.fs, 2))]]) #hz
    compressionC = parameter.compressionC
    
    #preallocazione dell'output: ha tante righe quante sono le bande
    #e tante colonne quante ne ha lo spettrogramma
    bandNoveltyCurves = np.zeros((bands.shape[0], specData.shape[1]))
    
    for band in np.arange(0, bands.shape[0]):
        
        bins = np.divide(np.round(bands[band,:]), np.divide(parameter.fs, parameter.win_len))
        bins = SetMaxIn1DArray(bins, 0)
        bins = SetMinIn1DArray(bins, np.round(parameter.win_len/2)+1)
        bins = np.vectorize(int)(bins)
        
        bandData = specData[bins[0]:bins[1],:]
        if parameter.logCompression and parameter.compressionC > 0:
            bandData = np.log(1 + np.multiply(bandData, compressionC))/np.log(1 + compressionC)
            
        diff_len = 0.3 #sec
        diff_len = max(np.ceil(diff_len*parameter.fs/parameter.stepsize),5)
        diff_len = 2*np.vectorize(MyRound)(np.divide(diff_len,2))+1
        aux_arr = np.array(-1*np.ones((int(np.floor(diff_len/2)),1)))
        aux_arr = np.append(aux_arr, 0)
        aux_arr = np.append(aux_arr, np.ones((int(np.floor(diff_len/2)),1)))
        diff_filter = np.multiply(handle(diff_len), aux_arr)
        aux_arr2 = np.transpose(np.matlib.repmat(bandData[:,0],int(np.floor(diff_len/2)),1))
        aux_arr2 = np.concatenate((aux_arr2, bandData), axis = 1)
        aux_arr2 = np.concatenate((aux_arr2, np.transpose(np.matlib.repmat(bandData[:,-1],int(np.floor(diff_len/2)),1))), axis = 1)
        diff_filter = diff_filter.reshape((diff_filter.shape[0],1))
        bandDiff = -1*conv2(aux_arr2, np.rot90(diff_filter,1), mode = 'same')
        bandDiff[bandDiff < 0] = 0
        bandDiff = bandDiff[:,int(np.floor(diff_len/2)-1):-int(np.floor(diff_len/2)+1)]
        
        #normalize band
        norm_len = 5
        norm_len = max(np.ceil(norm_len*parameter.fs/parameter.stepsize),3)
        norm_filter = handle(norm_len)
        aux_arr3 = (norm_filter/np.sum(norm_filter))
        aux_arr3 = aux_arr3.reshape((1,aux_arr3.shape[0]))
        norm_curve = conv2(np.sum(bandData, axis = 0).reshape((bandData.shape[1],1)), np.rot90(aux_arr3, 1), mode = 'same')
        #boundary correction
        norm_filter_sum = np.divide((np.sum(norm_filter)-np.cumsum(norm_filter)),np.sum(norm_filter))
        norm_filter_sum = norm_filter_sum.reshape((norm_filter_sum.shape[0], 1))
        bound = int(np.floor(norm_len/2))
        norm_curve[0:bound] = np.divide(norm_curve[0:bound], np.flipud(norm_filter_sum[0:bound].reshape((bound,1))))
        norm_curve[-bound:] = np.divide(norm_curve[-bound:], norm_filter_sum[0:bound])
        norm_curve = np.transpose(norm_curve)
        bandDiff = np.divide(bandDiff, norm_curve)

        noveltyCurve = np.sum(bandDiff, axis = 0)
        bandNoveltyCurves[band,:] = noveltyCurve
        
    noveltyCurve = np.mean(bandNoveltyCurves, axis = 0)

    #resample curve
    if parameter.resampleFeatureRate > 0 and not(parameter.featureRate == parameter.resampleFeatureRate):
        noveltyCurve, featureRate = resample_noveltyCurve(noveltyCurve, parameter)
        parameter.featureRate = featureRate
    
    noveltyCurve, local_average = novelty_smoothedSubtraction(noveltyCurve, parameter)
    
    return noveltyCurve, featureRate