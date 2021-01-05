import numpy as np

def minmax_scale(X:np.ndarray, axis=0, feature_range=(0,1)):
    X_out = np.zeros_like(X)
    X_min = np.min(X, axis=axis)
    X_max = np.max(X, axis=axis)
    
    X_out = (X - X_min) / (X_max - X_min)
    X_out = X_out * (feature_range[1]-feature_range[0]) + feature_range[0]
                          
    return X_out

def transpose(x:np.ndarray) -> np.ndarray:
    return x.transpose()