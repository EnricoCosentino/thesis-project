import numpy as np


def normalizeFeature(f_feature, normP, threshold):
    
    f_featureNorm = np.zeros(f_feature.shape[:], dtype = 'complex')
    
    unit_vec = np.ones((f_feature.shape[0], 1), dtype = 'complex')
    unit_vec = unit_vec/np.linalg.norm(unit_vec, normP)
    for k in np.arange(0, f_feature.shape[1]):
        n = np.linalg.norm(f_feature[:,k], normP)
        if n < threshold:
            f_featureNorm[:,k] = unit_vec
        else:
            f_featureNorm[:,k] = f_feature[:,k]/n
    
    return f_featureNorm