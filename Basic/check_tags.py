#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:53:39 2021

@author: arvid2
"""
import numpy as np

def check_tags(data, total_time, threshold_time, threshold_move):
    # Count how many occurences of every ID
    ID_counts = data["ID"].value_counts().to_frame("count")
    ID_counts.reset_index(inplace=True)
    ID_counts = ID_counts.rename(columns = {"index": "ID"})
    
    # IDs extraction
    activeID = ID_counts["ID"].values
    badID = ID_counts.loc[ID_counts["count"] < threshold_time*total_time]["ID"].values
    goodID = np.array(list(set(activeID) - set(badID)))
    
    # Calculate how much in y every activeID moved during the period
    individual_move = np.empty(len(activeID), dtype=int)
    for i in np.arange(0, len(activeID)):
        individual_df = data.loc[data["ID"] == activeID[i]]
        individual_move[i] = np.max(individual_df["y"].values) - np.min(individual_df["y"].values)
          
    stillID = activeID[individual_move < threshold_move]
        
    return activeID, badID, stillID, np.array(list(set(goodID) - set(stillID)))