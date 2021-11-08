import pandas as pd
import numpy as np

def fill_interval(data, start_time, end_time):
    data.reset_index(drop=True)
    newTimes = np.arange(start_time, end_time, 1)
    
    # Declare output dataframe
    filled_data = pd.DataFrame(columns = ["ID", "Time", "x", "y"])
    filled_data["Time"] = newTimes
    filled_data["ID"] = data["ID"].iloc[0]*np.ones(len(filled_data))
    
    # Remove duplicate time steps in input data
    no_duplicates = data.drop_duplicates(subset = "Time")
    
    # Find which time steps already exist in input data and their row numbers
    intersection = np.array(list(set(data["Time"].values).intersection(set(filled_data["Time"].values))))
    rows = np.where(np.in1d(filled_data["Time"].values, intersection))[0]
        
    # Insert rows of these timesteps into output object 
    filled_data.iloc[rows] = no_duplicates
    
    # Find first non nan x and y rows
    first_non_nan_x = filled_data["x"].first_valid_index()
    first_non_nan_y = filled_data["y"].first_valid_index()
    
    # Fill rows before first non nan value with that same value
    filled_data.loc[:first_non_nan_x, "x"] = filled_data["x"].iloc[first_non_nan_x]
    filled_data.loc[:first_non_nan_y, "y"] = filled_data["y"].iloc[first_non_nan_y]
    
    # # Fill nan values with makima interpolation...
  
    
    
    return filled_data
