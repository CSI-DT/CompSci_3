import numpy as np
import time
from numba import *

@njit
def d_calc(w1, w2, delta1, delta2):
    return (w1*delta1 + w2*delta2)/(w1 + w2) if (w1 + w2 != 0) else 0

# Returns interpolation evaluated at points xq based on points (x, v)
@njit
def makima(x, v, xq):
    h = np.diff(x)
    delta = np.diff(v) / h
    # time1 = time.perf_counter()


    n = len(delta) + 1 # Number of grid points
    padded_delta = np.empty(len(delta) + 4, dtype = float_)
    w1 = np.empty(n, dtype = float_)
    w2 = np.empty(n, dtype = float_)
    d = np.empty(n, dtype = float_)
    vq = np.empty(len(xq), dtype = float_)
    
    padded_delta[2:(2 + len(delta))] = delta
    
    # Pad delta with delta_i values outside boundary
    padded_delta[1] = 2*padded_delta[2] - padded_delta[3]
    padded_delta[0] = 2*padded_delta[1] - padded_delta[2]
    padded_delta[-2] = 2*padded_delta[-3] - padded_delta[-4]
    padded_delta[-1] = 2*padded_delta[-2] - padded_delta[-3]
    
    # Calculate weights and derivative ds
    for i in np.arange(2, n+2):
        w1[i-2] = abs(padded_delta[i+1] - padded_delta[i]) + 1/2 * abs(padded_delta[i+1] + padded_delta[i])
        w2[i-2] = abs(padded_delta[i-1] - padded_delta[i-2]) + 1/2 * abs(padded_delta[i-1] + padded_delta[i-2])
        d[i-2] = d_calc(w1[i-2], w2[i-2], padded_delta[i-1], padded_delta[i])
        
    # print("Slopes calc took ", time.perf_counter() - time1, " s")
    
    # Now we should find values vq for xq using the formula for 
    # polynomial Hermitian cubic interpolation 
    # time2 = time.perf_counter()
    for i in np.arange(0, len(xq) - len(xq[np.where(np.greater(xq, x[-1]))])): # Loop over all xq smaller than largest x
        low_x_index = 0    
        while x[low_x_index+1] < xq[i]:
            low_x_index += 1
        # low_x_index = bisect.bisect_left(x, xq[i]) - 1 # Find index of interval lower x
        hk = x[low_x_index + 1] - x[low_x_index] # Interval length
        s = xq[i] - x[low_x_index]         
        vq[i] = (3*hk*s**2 - 2*s**3)/(hk**3)*v[low_x_index + 1] + (hk**3 - 3*hk*s**2 + 2*s**3)/(hk**3)*v[low_x_index] + (s**2*(s - hk))/(hk**2)*d[low_x_index + 1] + (s*(s - hk)**2)/(hk**2) * d[low_x_index]
            
    # Set vq at xq > x[end] to v[end]        
    vq[np.where(np.greater(xq, x[-1]))] = v[-1] 
    # print("Values calc took ", time.perf_counter() - time2, " s")            
    return vq