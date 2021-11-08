def getInterval(individual_data, starttime, endtime):
    # Function to extract data for a time interval given data from a certain individual
    return individual_data.loc[(individual_data['time'] <= endtime) & (individual_data['time'] >= starttime)]