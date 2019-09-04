import numpy as np

from scipy.interpolate import interp1d

def rescaleTempoAxis(tempogram_in, BPM_in, BPM_out):
    if np.iscomplexobj(tempogram_in):
        r_in = np.real(tempogram_in)
        i_in = np.imag(tempogram_in)
        r_outf = interp1d(BPM_in, r_in, kind = 'nearest', axis = 0, bounds_error = False, fill_value = 0.)
        i_outf = interp1d(BPM_in, i_in, kind = 'nearest', axis = 0, bounds_error = False, fill_value = 0.)
        r_out = r_outf(BPM_out)
        i_out = i_outf(BPM_out)
        return np.vectorize(np.complex)(r_out, i_out), BPM_out
    
    f = interp1d(BPM_in, tempogram_in, kind = 'nearest', axis = 0, bounds_error = False, fill_value = 0.)
    return f(BPM_out), BPM_out