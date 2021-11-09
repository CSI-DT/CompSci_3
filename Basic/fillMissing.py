import pandas as pd
import numpy as np
import makimaInterpolator

# Fill missing data of a dataframe between start_time and end_time
def fillMissing(ind_data, start_time, end_time):
    ind_data.reset_index(drop=True)
    newTimes = np.arange(start_time, end_time+1, 1)

    # Declare output dataframe
    filled_data = pd.DataFrame(columns = ["ID", "Time", "x", "y"])
    filled_data["Time"] = newTimes
    filled_data["ID"] = ind_data["ID"].iloc[0]*np.ones(len(filled_data))
    
    # Remove duplicate time steps in input data
    no_duplicates = ind_data.drop_duplicates(subset = "Time")
    
    # Find which time steps already exist in input data and their row numbers
    intersection = np.array(list(set(no_duplicates["Time"].values).intersection(set(filled_data["Time"].values))))
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
    filled_data.iloc[filled_data["x"].dropna().to_frame().index]["x"].values
    filled_data.iloc[filled_data["y"].dropna().to_frame().index]["y"].values
    
    # non-nan x_times, y_times, values x, values y
    nnan_times_x, nnan_times_y, vx, vy = filled_data.iloc[filled_data["x"].dropna().to_frame().index]["Time"].values, filled_data.iloc[filled_data["y"].dropna().to_frame().index]["Time"].values, filled_data.iloc[filled_data["x"].dropna().to_frame().index]["x"].values, filled_data.iloc[filled_data["y"].dropna().to_frame().index]["y"].values
   
    # nan rows + times x and y
    nan_rows_x = filled_data[filled_data["x"].isnull().to_frame().any(axis=1)].index
    nan_rows_y = filled_data[filled_data["y"].isnull().to_frame().any(axis=1)].index
    nan_times_x = filled_data.loc[nan_rows_x, :]["Time"].values
    nan_times_y = filled_data.loc[nan_rows_y, :]["Time"].values
        
    # Interpolate
    vxq = makimaInterpolator.makima(nnan_times_x, vx, nan_times_x)    
    vyq = makimaInterpolator.makima(nnan_times_y, vy, nan_times_y)    
    
    filled_data.loc[nan_rows_x, 'x'] = vxq
    filled_data.loc[nan_rows_y, 'y'] = vyq
    
    
    return filled_data
