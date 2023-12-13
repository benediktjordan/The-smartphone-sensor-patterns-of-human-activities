#region import
import pandas as pd
import os
import time
import datetime as dt
import geopy.distance
import matplotlib.pyplot as plt

#endregion

#region import data & add CEST time column
df = pd.read_csv ('G:\\Meine Ablage\\MasterThesis @nlbb\\Iteration01\\I1_Data Collection\\Yaras Sprint\\20220523_Tobi_DataCollection4_locations.csv')
#create time column
def localtimefromunix(column):
    time = []
    for i in range(column.size):
        l = dt.datetime.fromtimestamp(column[i]/1000).strftime('%Y-%m-%d %H:%M:%S')
        time.append(l)
    return time

df["time"] = localtimefromunix(df["timestamp"])
#endregion

#region compute distance between points & assign them to clusters

#region trialout: create column which shows distance difference between the row and the row before

def distancetomeasurementbefore(column_latitude, column_longitude):
    distance_measbefore = [0]
    for i in range(column_latitude.size):
        if i == 0:
            continue
        coords_1 = (column_latitude[i], column_longitude[i])
        coords_2 = (column_latitude[i-1], column_longitude[i-1])
        distance = geopy.distance.geodesic(coords_1, coords_2).km*1000
        distance_measbefore.append(distance)
    return distance_measbefore

df["distance_measbefore"] = distancetomeasurementbefore(df["double_latitude"], df["double_longitude"])
#endregion

#region compute location "clusters"
def locationclusters(column_latitude, column_longitude):
    clusters = [[column_latitude[0], column_longitude[0]]]
    column_clusters_indices = []
    for i in range(column_latitude.size):
        coords_point = tuple((column_latitude[i], column_longitude[i]))
        for l in clusters:
            counter = 0
            coords_clust = tuple(l)
            distance_clustertopoint = geopy.distance.geodesic(coords_point, coords_clust).km*1000
            if distance_clustertopoint <= 20:
                column_clusters_indices.append(clusters.index(l))
                counter = counter+1
                break

        if counter == 0:
            clusters.append(list(coords_point))
            column_clusters_indices.append(clusters.index(list(coords_point)))

    return column_clusters_indices, clusters

column_clusters_indices, clusters = locationclusters(df["double_latitude"], df["double_longitude"])

df["location_clusters"] = column_clusters_indices
cluster_counts = df["location_clusters"].value_counts()
main_cluster = clusters[cluster_counts.index[0]]

bins = ("location 1", "location 2", "location 3", "location 4", "location 5")
plt.bar(bins, cluster_counts2["location_clusters"][0:5])
plt.show()
cluster_counts2 = cluster_counts.reset_index()

column_latitude = df["double_latitude"]
column_longitude = df["double_longitude"]
clusters = [[column_latitude[0], column_longitude[0]]]
clusters.append(0)
#endregion

