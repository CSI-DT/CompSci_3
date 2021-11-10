import pandas as pd
import time
import numpy as np
import itertools
from scipy.spatial import distance


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


def fillMissing(filled_data, start_time, end_time):
    # Find first non nan x and y rows
    first_non_nan_x = filled_data["x"].first_valid_index()
    first_non_nan_y = filled_data["y"].first_valid_index()

    # Fill rows before first non nan value with that same value
    filled_data.loc[:first_non_nan_x, "x"] = filled_data["x"].iloc[first_non_nan_x]
    filled_data.loc[:first_non_nan_y, "y"] = filled_data["y"].iloc[first_non_nan_y]

    # # Fill nan values with linear interpolation...
    filled_data = filled_data.astype(float).interpolate()

    return filled_data


def readFAfile(FAfile):
    var_names = ["FA", "id", "id1", "time", "x", "y", "z"]
    data_types = {'FA': object, 'id': np.uint64, 'id1': str, 'time': np.uint64, 'x': np.uint16, 'y': np.uint16,
                  'z': np.uint16}
    return pd.read_csv(FAfile, names=var_names, dtype=data_types)


if __name__ == '__main__':
    start_time = time.time()

    starttime = 1573862400
    endtime = 1573948799
    Total_time = endtime - starttime
    num = 86400
    times = np.arange(0, num) + starttime

    data = readFAfile('D:\cow_data\FA_20191116T000000UTC.csv')
    data = data.drop(columns=['FA', 'id1', 'z'])
    data['time'] = data['time'].round(-3) // 1000  # Round time to seconds from milliseconds

    # Some of the values get duplicated when we round to seconds. Remove them
    data = data.drop_duplicates(subset=['id', 'time'], keep='last')

    activeID, badID, stillID, movingID = checkTags(data, 86400, 0.3, 1800)

    LeftGroup, RightGroup = divideGroup(data, movingID, starttime, endtime)
    # IDlist = LeftGroup
    IDlist = movingID
    num_ids = len(IDlist)

    # data_complete is the set of all movingID's at all times
    data_complete = pd.DataFrame(np.array(
        [np.repeat(IDlist, 86400), np.resize(times, num_ids * 86400)]).T,
                                 dtype=np.uint64, columns=['id', 'time'])

    data = data_complete.merge(data, on=['id', 'time'], how='left')  # Padded data

    # Note: The data is sorted id wise, and time-wise for each id
    data = fillMissing(data, starttime, endtime)
    data = data.astype('int64')  # changes id,time,x,y to int64

    # save(filledIndividualIinterval_FileName, 'Filled_individual_interval');

    # Distance matrix
    Pair = np.array(list(itertools.combinations(np.arange(num_ids), 2)))  # Find all combinations
    Distance = np.empty(shape=(86400, len(Pair)))
    for i in range(len(Pair)):
        P1 = np.array([data['x'][Pair[i][0] * 86400:(Pair[i][0] + 1) * 86400],
                   data['y'][Pair[i][0] * 86400:(Pair[i][0] + 1) * 86400]]).T
        P2 = np.array([data['x'][Pair[i][1] * 86400:(Pair[i][1] + 1) * 86400],
                   data['y'][Pair[i][1] * 86400:(Pair[i][1] + 1) * 86400]]).T
        Distance[:, i] = np.sqrt(np.einsum('ij,ij->i', P1-P2, P1-P2))

    print("--- %s seconds ---" % (time.time() - start_time))
