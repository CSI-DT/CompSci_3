import numpy as np
import bisect

# Calculates derivatives d of makima formula 
def makimaSlopes(delta):
    n = len(delta) + 1 # Number of grid points
    padded_delta = np.empty(len(delta) + 4, dtype = float)
    padded_delta[2:(2 + len(delta))] = delta
    
    # Pad delta with delta_i values outside boundary
    padded_delta[1] = 2*padded_delta[2] - padded_delta[3]
    padded_delta[0] = 2*padded_delta[1] - padded_delta[2]
    padded_delta[-2] = 2*padded_delta[-3] - padded_delta[-4]
    padded_delta[-1] = 2*padded_delta[-2] - padded_delta[-3]
    
    # Calculate weights and derivative ds
    w1 = np.empty(n, dtype = float)
    w2 = np.empty(n, dtype = float)
    d = np.empty(n, dtype = float)
    for i in np.arange(2, n+2):
        w1[i-2] = abs(padded_delta[i+1] - padded_delta[i]) + 1/2 * abs(padded_delta[i+1] + padded_delta[i])
        w2[i-2] = abs(padded_delta[i-1] - padded_delta[i-2]) + 1/2 * abs(padded_delta[i-1] + padded_delta[i-2])
        d[i-2] = w1[i-2]/(w1[i-2] + w2[i-2]) * padded_delta[i-1] + w2[i-2]/(w1[i-2] + w2[i-2]) * padded_delta[i]
    
    # Replace nan values of d with 0. Happens if five consequtive constant values
    # in v
    np.nan_to_num(d, copy=False)

    return d
    
# Returns interpolation evaluated at points xq based on points (x, v)
def makima(x, v, xq):
    h = np.diff(x)
    delta = np.diff(v) / h
    slopes = makimaSlopes(delta) # d vector from internet
    vq = np.empty(len(xq), dtype = float)
    # Now we should find values vq for xq using the formula for 
    # polynomial Hermitian cubic interpolation 
    # Loop through xq, find in between which x values it exists
    # find the vq of xq using the formula with the two padding x values
    # maybe could handle endpoints better!
    for i in np.arange(0, len(xq)):
        low_x_index = bisect.bisect_left(x, xq[i]) - 1 # Find index of interval lower x
        if (low_x_index != len(x)-1): # If the xq is not outside x
            hk = x[low_x_index + 1] - x[low_x_index] # Interval length
            s = xq[i] - x[low_x_index]         
            vq[i] = (3*hk*s**2 - 2*s**3)/(hk**3)*v[low_x_index + 1] + (hk**3 - 3*hk*s**2 + 2*s**3)/(hk**3)*v[low_x_index] + (s**2*(s - hk))/(hk**2)*slopes[low_x_index + 1] + (s*(s - hk)**2)/(hk**2) * slopes[low_x_index]
        else: # If xq is outside x, set as ending value of data
            vq[i] = v[-1]

    return vq

    
    
    
    
    
    
    
    
    
    
    
    
    
    