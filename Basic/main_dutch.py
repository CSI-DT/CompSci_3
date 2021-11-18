import pandas as pd
import time
import numpy as np
import makimaInterpolator as ma
import itertools
from scipy.spatial import distance
import pickle
import datetime

def getIndividual(indata, id):
    # Function to extract all the data given a certain individual id
    return indata.loc[indata['id'] == id]


def getInterval(individual_data, starttime, endtime):
    # Function to extract data for a time interval given data from a certain individual
    return individual_data.loc[(individual_data['time'] <= endtime) & (individual_data['time'] >= starttime)]

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


def checkTags(in_data, total_time, threshold_time, threshold_move):
    # Count how many occurences of every ID
    ID_counts = in_data["id"].value_counts().to_frame("count")
    ID_counts.reset_index(inplace=True)
    ID_counts = ID_counts.rename(columns={"index": "id"})

    # IDs extraction
    activeID = in_data['id'].unique()
    badID = ID_counts.loc[ID_counts["count"] < threshold_time * total_time]["id"].values
    goodID = np.array(list(set(activeID) - set(badID)))

    # Calculate how much in y every activeID moved during the period
    individual_move = np.empty(len(activeID), dtype=int)

    for i in range(len(activeID)):
        individual_data = getIndividual(in_data, activeID[i])
        individual_move[i] = np.max(individual_data["y"].values) - np.min(individual_data["y"].values)

    stillID = activeID[individual_move < threshold_move]

    return activeID, badID, stillID, np.array(list(set(goodID) - set(stillID)))

# Alexander's pimped out version 2
def fillMissing(ind_data, start_time, end_time):
    # time1 = time.perf_counter()
    ind_data.reset_index(drop=True)
    newTimes = np.arange(start_time, end_time + 1, 1)
    total_time = end_time - start_time + 1

    # Declare output dataframe
    filled_data = pd.DataFrame(np.array(
        [ind_data["id"].iloc[0] * np.ones(total_time), newTimes, np.full(total_time, np.nan),
         np.full(total_time, np.nan)]).T, columns=["id", "time", "x", "y"])
    # Remove duplicate time steps in input data
    no_duplicates = ind_data.groupby(['id', 'time']).mean().reset_index()
    # no_duplicates = ind_data # Duplicates already removed in social_contact

    # Find which time steps already exist in input data and their row numbers
    intersection = np.array(list(set(no_duplicates["time"].values).intersection(set(filled_data["time"].values))))
    rows = np.where(np.in1d(filled_data["time"].values, intersection))[0]
    # Insert rows of these timesteps into output object
    filled_data.iloc[rows] = no_duplicates

    filling_values = (ind_data[ind_data['id'] == filled_data.iloc[0].id]).iloc[0]
    idx = filling_values.time - start_time
    for i in range(int(idx)):
        filled_data.at[i, 'x'] = filling_values['x']
        filled_data.at[i, 'y'] = filling_values['y']

    # non-nan x_times, y_times, values x, values y
    nnan_times_x, nnan_times_y, vx, vy = filled_data.iloc[filled_data["x"].dropna().to_frame().index]["time"].values, filled_data.iloc[filled_data["y"].dropna().to_frame().index]["time"].values, filled_data.iloc[filled_data["x"].dropna().to_frame().index]["x"].values, filled_data.iloc[filled_data["y"].dropna().to_frame().index]["y"].values

    # nan rows + times x and y
    nan_rows_x = filled_data[filled_data["x"].isnull().to_frame().any(axis=1)].index
    nan_rows_y = filled_data[filled_data["y"].isnull().to_frame().any(axis=1)].index
    nan_times_x = filled_data.loc[nan_rows_x, :]["time"].values
    nan_times_y = filled_data.loc[nan_rows_y, :]["time"].values

    # time2 = time.perf_counter()
    # Interpolate
    # makima
    filled_data.loc[nan_rows_x, 'x'] = ma.makima(nnan_times_x, vx, nan_times_x)
    filled_data.loc[nan_rows_y, 'y'] = ma.makima(nnan_times_y, vy, nan_times_y)
    
    # linear
    # filled_data = filled_data.astype(float).interpolate() 
    # end_time = time.perf_counter()

    # print("Interpolation took ", (end_time - time2), "s or ", (end_time - time2)/(end_time - time1), " of total time.")
    return filled_data

def removeDryArea(in_data, idlist): 
    threshold = 8000 
    mean_y = np.empty(len(idlist)) 
    for i in range(len(idlist)): 
        individual_data = getIndividual(in_data, idlist[i]) 
        mean_y[i] = 0.5 * (individual_data['y'].max() + individual_data['y'].min()) 
        DryAreaValue = np.where(mean_y > threshold) 
        MainAreaValue = np.where(mean_y < threshold) 
        return idlist[MainAreaValue], idlist[DryAreaValue]

def readFAfile(FAfile):
    var_names = ["FA", "id", "id1", "time", "x", "y", "z"]
    data_types = {'FA': object, 'id': np.int64, 'id1': str, 'time': np.int64, 'x': np.float64, 'y': np.float64,
                  'z': np.float64}
    return pd.read_csv(FAfile, names=var_names, dtype=data_types)

if __name__ == '__main__':
    #'''
    tic = time.time()

    DistThreshold = 250
    realContactThreshold = 600
    barn = np.arange(1,26)
    cubic = np.arange(1, 15)
    feed = np.arange(15, 26)

    tic1 = time.time()
    file_name = 'FA_20201016T000000UTC.csv'
    year, month, day = int(file_name[3:7]), int(file_name[7:9]), int(file_name[9:11])
    starttime = int(datetime.datetime(year, month, day).timestamp())
    endtime = starttime + 24*60*60 - 1   
    Total_time = endtime - starttime + 1
    
    data = readFAfile(file_name)
    data = data.drop(columns=['FA', 'id1', 'z'])
    data['time'] = data['time'] // 1000 # Round time to seconds from milliseconds
    data = data[data.time <= endtime] # Remove rows outside time scope
    data = data[data.time >= starttime ]
    print("Reading FA file: %s" % (time.time() - tic1))
    
    a = pd.read_csv('a_dutch.csv', header = None)
    tic2 = time.time()
    activeID, badID, stillID, movingID = checkTags(data, Total_time, 0.3, 1800)
    
    print("Check tags: %s" % (time.time() - tic2))
    tic2 = time.time()
    mainAreaID, dryAreaID = removeDryArea(data, movingID)
   
    # LeftGroup, RightGroup = divideGroup(data, movingID, starttime, endtime)
    #IDlist = movingID
    IDlist = mainAreaID
    IDlist.sort()
    
    print("Remove dry area: %s " % (time.time() - tic2))
    data = data.sort_values(by='time').reset_index(drop=True)
            
    tic2 = time.time()
    
    # data = data[data['id'].isin(IDlist)] IDlist.sort() 
    # data = fillMissing(data, IDlist, starttime, endtime) 
    # Makima version
    # Problem eftersom max time i data är 1573948800 inte 799
    filled_individuals = np.empty(len(IDlist), dtype = type(data))
    for i in np.arange(0, len(IDlist)):
        filled_individuals[i] = fillMissing(getIndividual(data, IDlist[i]), starttime, endtime)
    data = pd.concat(filled_individuals)

    data.loc[data['x'] > 2991, 'x'] = 2991
    data.loc[data['y'] > 7666, 'y'] = 7666
    data.loc[data['y'] < 718, 'y'] = 718
    
    # data = fillMissing(data, IDlist, starttime, endtime)
    
    print("Fill data: %s" % (time.time() - tic2))
    # Distance matrix
    tic2 = time.time()
    Pair = np.array(list(itertools.combinations(np.arange(len(IDlist)), 2)))  # Find all combinations
    Distance = np.empty(Total_time, dtype = float)
    all_contacts = np.empty(len(barn), dtype = object)
    real_contact_list_cubic = np.empty(len(Pair), dtype = object)
    real_contact_list_feed = np.empty(len(Pair), dtype = object)
    
    # Contact calculations
    for i in range(len(Pair)):
        tic4 = time.perf_counter()
        
        x1 = np.array([data['x'][Pair[i][0] * Total_time:(Pair[i][0] + 1) * Total_time]])
        y1 = np.array([data['y'][Pair[i][0] * Total_time:(Pair[i][0] + 1) * Total_time]])
        x2 = np.array([data['x'][Pair[i][1] * Total_time:(Pair[i][1] + 1) * Total_time]])
        y2 = np.array([data['y'][Pair[i][1] * Total_time:(Pair[i][1] + 1) * Total_time]])
        dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        
        in_range = dist < DistThreshold 
        if in_range.sum() >= realContactThreshold: 
            for j in range(len(barn)):
                all_contacts[j] = np.where(((x1 >= a[0][j]) & (x1 < a[1][j]) & (y1 >= a[2][j]) & (y1 < a[3][j]) & (in_range)))[1]

            cubic_contacts = np.concatenate(all_contacts[cubic - 1])
            feed_contacts = np.concatenate(all_contacts[feed - 1])
            
            cubic_area = []
            feed_area = []
            for j in cubic:
                cubic_area = np.concatenate([cubic_area, j*np.ones(len(all_contacts[j-1]))])
            
            for j in feed:
                feed_area = np.concatenate([feed_area, j*np.ones(len(all_contacts[j-1]))])
            
            if (len(cubic_contacts) >= realContactThreshold):
                real_contact_list_cubic[i] = np.empty([len(cubic_contacts), 7], dtype = float)
                real_contact_list_cubic[i][:, 0] = np.reshape(cubic_contacts, [len(cubic_contacts), ])
                real_contact_list_cubic[i][:, 1] = np.reshape(x1.T[cubic_contacts], [len(cubic_contacts), ])
                real_contact_list_cubic[i][:, 2] = np.reshape(y1.T[cubic_contacts], [len(cubic_contacts), ])
                real_contact_list_cubic[i][:, 3] = np.reshape(x2.T[cubic_contacts], [len(cubic_contacts), ])
                real_contact_list_cubic[i][:, 4] = np.reshape(y2.T[cubic_contacts], [len(cubic_contacts), ])
                real_contact_list_cubic[i][:, 5] = np.reshape(dist.T[cubic_contacts], [len(cubic_contacts), ])
                real_contact_list_cubic[i][:, 6] = np.reshape(cubic_area, [len(cubic_contacts), ])
                                
            if(len(feed_contacts) >= realContactThreshold):
                real_contact_list_feed[i] = np.empty([len(feed_contacts), 7], dtype = float)
                real_contact_list_feed[i][:, 0] = np.reshape(feed_contacts, [len(feed_contacts), ])
                real_contact_list_feed[i][:, 1] = np.reshape(x1.T[feed_contacts], [len(feed_contacts), ])
                real_contact_list_feed[i][:, 2] = np.reshape(y1.T[feed_contacts], [len(feed_contacts), ])
                real_contact_list_feed[i][:, 3] = np.reshape(x2.T[feed_contacts], [len(feed_contacts), ])
                real_contact_list_feed[i][:, 4] = np.reshape(y2.T[feed_contacts], [len(feed_contacts), ])
                real_contact_list_feed[i][:, 5] = np.reshape(dist.T[feed_contacts], [len(feed_contacts), ])
                real_contact_list_feed[i][:, 6] = np.reshape(feed_area, [len(feed_contacts), ])
                
    duration_whole = np.zeros(len(IDlist), dtype = int)
    friends_whole = np.zeros(len(IDlist), dtype = int)
    duration_cubic = np.zeros(len(IDlist), dtype = int)
    friends_cubic = np.zeros(len(IDlist), dtype = int)
    duration_feed = np.zeros(len(IDlist), dtype = int)
    friends_feed = np.zeros(len(IDlist), dtype = int)
    for i in np.arange(0, len(Pair)):
        i1 = Pair[i, 0]
        i2 = Pair[i, 1]     
        
        duration_cubic[i1] += len(real_contact_list_cubic[i]) if real_contact_list_cubic[i] is not None else 0
        duration_cubic[i2] += len(real_contact_list_cubic[i]) if real_contact_list_cubic[i] is not None else 0
        if real_contact_list_cubic[i] is not None:
           if len(real_contact_list_cubic[i]) > 0:
                friends_cubic[i1] += 1
                friends_cubic[i2] += 1
            
        duration_feed[i1] += len(real_contact_list_feed[i]) if real_contact_list_feed[i] is not None else 0
        duration_feed[i2] += len(real_contact_list_feed[i]) if real_contact_list_feed[i] is not None else 0
        if real_contact_list_feed[i] is not None:
            if len(real_contact_list_feed[i]) > 0:
                friends_feed[i1] += 1
                friends_feed[i2] += 1
            
    contact_table = np.reshape(np.concatenate([IDlist, duration_cubic, friends_cubic, duration_feed, friends_feed]), [len(IDlist), 5], order = 'F')
    print("Contact calculations: %s" % (time.time() - tic2))
    
    np.savetxt("contact_table.csv", contact_table, delimiter=',')
    with open('data.pkl', 'wb') as f:
        pickle.dump([real_contact_list_cubic, real_contact_list_feed], f, protocol = -1)
   
    
    print("Total time:  %s" % (time.time() - tic))

    # # '''
