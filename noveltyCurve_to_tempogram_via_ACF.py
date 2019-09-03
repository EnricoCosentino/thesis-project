import numpy as np

from warnings import warn
from statsmodels.tsa.stattools import ccf, acf
def MyRound(number):
        return round(number + 0.0001)

#Da risultati estremamente diversi rispetto a entrambe le funzioni di libreria
#e rispetto a MATLAB
def UnbiasedAutocorrelation(f, N = 0):
    if N == 0:
        N = len(f)
    fvi = np.fft.fft(f, n = 2*N)
    acf = np.real(np.fft.fft(fvi * np.conjugate(fvi))[:N])
    acf = acf/(N - np.arange(N))
    return acf

def noveltyCurve_to_tempogram_via_ACF(novelty, parameter = lambda:0):
    if (not hasattr(parameter,'featureRate')):
        warn('featureRate unset! Assuming 100!')
        parameter.featureRate = 100
    if (not hasattr(parameter,'tempoWindow')):
        parameter.tempoWindow = 6 #sec
    if (not hasattr(parameter,'stepsize')):
        parameter.stepsize = np.ceil(parameter.featureRate/5) # in frames (featureRate)
    if (not hasattr(parameter,'maxLag')):
        parameter.maxLag = 60/30 # 2 sec, corresponding to 30 bpm
    if (not hasattr(parameter,'minLag')):
        parameter.minLag = 60/600 # 0.1 sec, corresponding to 600 bpm
    if (not hasattr(parameter,'normalization')):
        parameter.normalization = 'unbiasedcoeff'
        
    win_len = MyRound(np.multiply(parameter.tempoWindow, parameter.featureRate))
    win_len += np.mod(win_len,2) - 1
    stepsize = parameter.stepsize
    maxLag = np.ceil(np.multiply(parameter.maxLag, parameter.featureRate))
    minLag = np.floor(np.multiply(parameter.minLag, parameter.featureRate)) + 1
    
    aux_arr = np.zeros(int(MyRound(win_len/2)))
    aux_arr = np.append(aux_arr, novelty)
    aux_arr = np.append(aux_arr, np.zeros(int(MyRound(win_len/2))))
    noveltyPadded = aux_arr
    num_win = int(np.fix((noveltyPadded.shape[0]-win_len+stepsize)/stepsize))
    
    noveltyNorm = np.zeros(num_win)
    N = np.zeros(num_win)    
    tempogram = np.zeros((int(maxLag), num_win))
    
    for win in range(0, num_win):
        start = int(max(1, np.floor(win*stepsize + 1)))
        stop = int(min(noveltyPadded.shape[0], np.ceil(start + win_len - 1)))
        maxL = int(min(maxLag, stop - start))
        window = np.ones(stop - start + 1)

        #AUTOCORRELATION
        nov = noveltyPadded[start-1:stop]
        if parameter.normalization == 'unbiasedcoeff':
            #Risultati molto simili all'originale
            xcr = np.correlate(np.multiply(window,nov),np.multiply(window,nov), 'full')[len(nov)-1-maxL:len(nov)+maxL]
            xcr = np.divide(xcr, xcr[maxL])
            xcr = xcr[maxL+1:]
            
            #Risultati simili all'originale ma non quanto np.correlate
            xcr1 = acf(np.multiply(window,nov), nlags = maxL, unbiased = True, fft = 'false')
            xcr1 = xcr1[1:]
            
            #Risultati molto diversi all'inizio, poi la differenza diventa minore della altre implementazioni
            xcr2 = ccf(np.multiply(window,nov), np.multiply(window,nov), unbiased = True)[int(len(nov)/2)-maxL-1:int(len(nov)/2)+maxL]
            xcr2 = xcr2[maxL+1:]
            
            #Differenza superiore alle altre di 4 ordini di grandezza, aumenta man mano che si va avanti
            xcr3 = UnbiasedAutocorrelation(np.multiply(window,nov))[len(nov)-1-maxL:len(nov)+maxL]
            xcr3 = np.divide(xcr3, xcr3[0])
            xcr3 = xcr3[1:]
        #I risultati differiscono considerevolmente 
        tempogram[:,win] = np.append(xcr, np.zeros(int(maxLag)-maxL))
        
        noveltyNorm[win] = np.sum(np.power(np.multiply(window,nov),2))
        N[win] = stop-start+1
    
        tempogram = np.flipud(tempogram[int(minLag)-1:,:])
        
        T = np.divide(np.arange(0, num_win), parameter.featureRate/parameter.stepsize)
        Lag = np.flipud(np.divide(np.arange(minLag, maxLag+1, 1), parameter.featureRate))
