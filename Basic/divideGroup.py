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