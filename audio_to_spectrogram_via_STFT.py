import numpy as np

#Makes Python rounding act like MATLAB's
def MyRound(number):
        return round(number + 0.0001)

def audio_to_spectrogram_via_STFT(f_audio, parameter = lambda:0):

    handle = lambda n: 0.5-0.5*np.cos(2*np.pi*(np.transpose(np.arange(0, n))/(n-1)))

    #Default inizialization of parameter
    if (not hasattr(parameter, 'StftWindow')): #Non controllo se e 0 perche e un array e su Python crea problemi
        parameter.StftWindow = handle(4096)
    windowLength = len(parameter.StftWindow)
    if (not hasattr(parameter, 'stepsize')) or parameter.stepsize == 0:
        parameter.stepsize = MyRound(windowLength/2)
    if (not hasattr(parameter, 'nFFT')) or parameter.nFFT == 0:
        parameter.nFFT = windowLength
    if (not hasattr(parameter, 'returnMagSpec')) or parameter.returnMagSpec == 0:
        parameter.returnMagSpec = 0
    if (not hasattr(parameter, 'coefficientRange')) or parameter.coefficientRange == 0:
        parameter.coefficientRange = [1, int(np.floor(max(parameter.nFFT, windowLength)/2)+1)]
    if (not hasattr(parameter, 'fs')) or parameter.fs == 0:
        parameter.fs = 22050

    #Pre calculations
    stepsize = parameter.stepsize
    featureRate = parameter.fs/(stepsize)
    wav_size = len(f_audio)
    win = parameter.StftWindow
    first_win = np.floor(windowLength/2)
    num_frames = np.ceil(wav_size/stepsize)
    num_coeffs = parameter.coefficientRange[-1] - parameter.coefficientRange[0]+1
    zerosToPad = max(0, parameter.nFFT - windowLength)

    #Memory allocation
    f_spec = np.zeros((int(num_coeffs),int(num_frames)))
    comp = np.vectorize(complex)
    if parameter.returnMagSpec == 0:
        f_spec = comp(f_spec)

    #First window center is at 0 seconds
    #frame is an array of indexes
    frame = np.arange(0, windowLength)
    frame = frame - first_win + 1
    for n in np.arange(0, num_frames):

        numZeros = sum(1 for i in frame if i < 1)
        numVals = sum(1 for i in frame if i > 0)
        
        if numZeros > 0:
            x = np.append(np.zeros((numZeros,1)),f_audio[:numVals])
            x = np.multiply(x, win)
        elif frame[-1] > wav_size:
            x = np.append(f_audio[int(frame[0]):wav_size], np.zeros((windowLength - (wav_size-int(frame[0])),1)))
            x = np.multiply(x, win)
        else:
            x = np.multiply(f_audio[frame.astype(int)], win)
            
        if zerosToPad > 0:
            x = np.append(x, np.zeros((zerosToPad,1)))
        
        Xs = np.fft.fft(x) #Piu ci si avvicina alla meta piu i risultati divergono da quelli MATLAB
        if parameter.returnMagSpec:
            f_spec[:,int(n)] = np.abs(Xs[parameter.coefficientRange[0]-1:parameter.coefficientRange[-1]])
        else:
            f_spec[:,int(n)] = Xs[parameter.coefficientRange[0]-1:parameter.coefficientRange[-1]]

        frame = frame + stepsize
        
    t = np.arange(0, f_spec.shape[1]) * stepsize/parameter.fs
    f = np.transpose(np.arange(0, np.floor(max(parameter.nFFT, windowLength)/2)+1)) / np.floor(max(parameter.nFFT, windowLength)/2) * (parameter.fs/2)
    f = f[parameter.coefficientRange[0]:parameter.coefficientRange[-1]]
    
    return f_spec, featureRate,
    #return f_spec, featureRate, f, t