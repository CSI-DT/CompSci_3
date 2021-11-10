import numpy as np
import pandas as pd
import makimaInterpolator
import time

def readFAfile(FAfile):
    var_names = ["FA", "id", "id1", "time", "x", "y", "z"]
    data_types = {'FA': object, 'id': np.uint64, 'id1': str, 'time': np.uint64, 'x': np.uint16, 'y': np.uint16,
                  'z': np.uint16}
    return pd.read_csv(FAfile, names=var_names, dtype=data_types)

def getInterval(individual_data, starttime, endtime):
    # Function to extract data for a time interval given data from a certain individual
    return individual_data.loc[(individual_data['time'] <= endtime) & (individual_data['time'] >= starttime)]

def getIndividual(indata, id):
    # Function to extract all the data given a certain individual id
    return indata.loc[indata['id'] == id]


# Takes pandas dataframe with measurement data and outputs 4 arrays 
# of IDs.
# Input: pandas dataframe with columns ID, time, x, y
# Outputs:
#   activeID - array of all unique IDs in data
#   badID - array of all IDs with low measurement count
#   stillID - array of all IDs of not moving tags
#   movingID - (activeID - badID - stillID)
def checkTags(data, total_time, threshold_time, threshold_move):
    # Count how many occurences of every ID
    ID_counts = data["id"].value_counts().to_frame("count")
    ID_counts.reset_index(inplace=True)
    ID_counts = ID_counts.rename(columns = {"index": "id"})
    
    # IDs extraction
    activeID = ID_counts["id"].values
    badID = ID_counts.loc[ID_counts["count"] < threshold_time*total_time]["id"].values
    goodID = np.array(list(set(activeID) - set(badID)))
    
    # Calculate how much in y every activeID moved during the period
    individual_move = np.empty(len(activeID), dtype=int)
    for i in np.arange(0, len(activeID)):
        individual_df = data.loc[data["id"] == activeID[i]]
        individual_move[i] = np.max(individual_df["y"].values) - np.min(individual_df["y"].values)
          
    stillID = activeID[individual_move < threshold_move]
        
    return activeID, badID, stillID, np.array(list(set(goodID) - set(stillID)))

# Fill missing data of a dataframe between start_time and end_time
def fillMissing(ind_data, start_time, end_time):
    # time1 = time.perf_counter()
    ind_data.reset_index(drop=True)
    newTimes = np.arange(start_time, end_time+1, 1)

    # Declare output dataframe
    filled_data = pd.DataFrame(columns = ["id", "time", "x", "y"])
    filled_data["time"] = newTimes
    filled_data["id"] = ind_data["id"].iloc[0]*np.ones(len(filled_data))
    
    # Remove duplicate time steps in input data
    no_duplicates = ind_data.drop_duplicates(subset = "time")
    # no_duplicates = ind_data # Duplicates already removed in social_contact
    
    # Find which time steps already exist in input data and their row numbers
    intersection = np.array(list(set(no_duplicates["time"].values).intersection(set(filled_data["time"].values))))
    rows = np.where(np.in1d(filled_data["time"].values, intersection))[0]
        
    # Insert rows of these timesteps into output object 
    filled_data.iloc[rows] = no_duplicates
    
    # Find first non nan x and y rows
    first_non_nan_x = filled_data["x"].first_valid_index()
    first_non_nan_y = filled_data["y"].first_valid_index()
    
    # Fill rows before first non nan value with that same value
    filled_data.loc[:first_non_nan_x, "x"] = filled_data["x"].iloc[first_non_nan_x]
    filled_data.loc[:first_non_nan_y, "y"] = filled_data["y"].iloc[first_non_nan_y]
    
    # non-nan x_times, y_times, values x, values y
    nnan_times_x, nnan_times_y, vx, vy = filled_data.iloc[filled_data["x"].dropna().to_frame().index]["time"].values, filled_data.iloc[filled_data["y"].dropna().to_frame().index]["time"].values, filled_data.iloc[filled_data["x"].dropna().to_frame().index]["x"].values, filled_data.iloc[filled_data["y"].dropna().to_frame().index]["y"].values
   
    # nan rows + times x and y
    nan_rows_x = filled_data[filled_data["x"].isnull().to_frame().any(axis=1)].index
    nan_rows_y = filled_data[filled_data["y"].isnull().to_frame().any(axis=1)].index
    nan_times_x = filled_data.loc[nan_rows_x, :]["time"].values
    nan_times_y = filled_data.loc[nan_rows_y, :]["time"].values
        
    # Interpolate
    # time2 = time.perf_counter()
    filled_data.loc[nan_rows_x, 'x'] = makimaInterpolator.makima(nnan_times_x, vx, nan_times_x)    
    filled_data.loc[nan_rows_y, 'y'] = makimaInterpolator.makima(nnan_times_y, vy, nan_times_y)    
           
    # end_time = time.perf_counter()
    
    # print("Interpolation took ", (end_time - time2), "s or ", (end_time - time2)/(end_time - time1), " of total time.")    
    return filled_data

def divideGroup(data, IDlist, starttime, endtime):
    # Divide group for swedish farm.
    newStarttime = 1573898400  # 10:00:00 in posixtime
    newEndtime = 1573912800  # 14:00:00 in posixtime
    Group = np.ones(len(IDlist))
    for ID in range(len(IDlist)):
        individual_interval = getInterval(getIndividual(data, IDlist[ID]), newStarttime, newEndtime)
        if (individual_interval['x'].mean() > 1670):
            Group[ID] = 2

    LeftGroup = []
    RightGroup = []
    for i in range(len(IDlist)):
        if (Group[i] == 1):
            LeftGroup.append(IDlist[i])
        else:
            RightGroup.append(IDlist[i])

    return LeftGroup, RightGroup















