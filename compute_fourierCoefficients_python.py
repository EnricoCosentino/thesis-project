import numpy as np

def compute_fourierCoefficients_python(s, win, noverlap, f, fs = 1):
    
    win_len = len(win)
    hopsize = win_len - noverlap

    T = np.divide(np.arange(0, win_len), fs)
    win_num = int(np.fix((len(s) - noverlap)/(win_len-noverlap)))
    x = np.zeros((win_num, len(f)), dtype = np.complex)
    t = np.arange(win_len/2, len(s)-win_len/2, hopsize)/fs

    twoPiT = 2*np.pi*T
    
    for f0 in np.arange(0, len(f)):
        
        twoPiFt = f[f0]*twoPiT
        cosine = np.cos(twoPiFt)
        sine = np.sin(twoPiFt)
        
        for w in np.arange(0, win_num):
            start = w*hopsize+1
            stop = start + win_len - 1
            
            sig = np.multiply(s[int(start-1):int(stop)], win)
            co = np.sum(np.multiply(sig, cosine))
            si = np.sum(np.multiply(sig, sine))
            x[w, f0] = np.complex(co, si)

    x = np.transpose(x)
    return x, f, t