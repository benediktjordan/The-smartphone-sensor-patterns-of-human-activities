#region import
import pandas as pd
import os
import time
import datetime as dt
import geopy.distance
import matplotlib.pyplot as plt
import numpy as np
#endregion

#region import data & add CEST time column
df = pd.read_csv ('G:\\Meine Ablage\\MasterThesis @nlbb\\SmartphoneActivityDetection\\Dataset\\Davidello Data-20220606T232545Z-001\\Davidello Data\\20220525_Instagram Tina + Lea\\20220525_Instagram_Tina.csv')

#create time column
def localtimefromunix(column):
    time = []
    for i in range(column.size):
        if np.isnan(column[i]) == True:
            time.append("nan")
        else:
            l = dt.datetime.fromtimestamp(column[i]).strftime('%Y-%m-%d %H:%M:%S')
            time.append(l)
    return time

df["startTime"] = localtimefromunix(df["startTimestamp"])
df["finishedTime"] = localtimefromunix(df["finishedTimestamp"])
df["lastIOTime"] = localtimefromunix(df["lastIOTimestamp"])
#endregion

#region select only relevant rows & include labels
df_labels = df = pd.read_csv ('G:\\Meine Ablage\\MasterThesis @nlbb\\SmartphoneActivityDetection\\Dataset\\Internal Davidello Data Collection Database Instagram.csv')

def loadlabels(csv, date, name_starttime, name_endtime, column_label, original_data):
    column_starttime = csv[date]+ " " + csv[name_starttime]
    column_endtime = csv["date"] + csv["name_endtime"]
    for i in range(csv["date"].size):
        starttime = dt.datetime.strptime(column_starttime[i],'%B %d, %Y %H:%M')
        endtime = dt.datetime.strptime(column_endtime[i],'%B %d, %Y %H:%M')


## NOTE: 07.06.2022 - this is work in progress; didnt manage to finish it yet

#endregion

#region manually label
df_labels = df = pd.read_csv ('G:\\Meine Ablage\\MasterThesis @nlbb\\SmartphoneActivityDetection\\Dataset\\Internal Davidello Data Collection Database Instagram.csv')

df.drop(df.loc[0:35].index, inplace = True)

df.drop(df.loc[228:246].index, inplace = True)

df.drop(df.loc[467:485].index, inplace = True)

df.drop(df.loc[784:6909].index, inplace = True)

label = ["Writing DM"]*192+["Stories"]*220 + ["Reels"]*298

df["activity"] = label