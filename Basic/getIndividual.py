def getIndividual(indata, id):
    # Function to extract all the data given a certain individual id
    return indata.loc[indata['id'] == id]