def readFAfile(FAfile):
    var_names = ["FA", "id", "id1", "time", "x", "y", "z"]
    # data_types = []
    return pd.read_csv(FAfile, names=var_names)