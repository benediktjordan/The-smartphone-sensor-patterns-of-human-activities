#region import
import pandas as pd
import json
from datetime import datetime
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import os
from tqdm import tqdm
# computing distance
import geopy.distance
import time

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


#endregion

# create clustering object
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html


# load data
#complete data
## only small chunk from csv file
df = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/locations_all.csv", nrows=200000)
#df = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/locations_all.csv")

#dir_databases = "/Volumes/INTENSO/In Usage new/Databases"
#merge_locations = Merge_Transform(dir_databases, "locations")
#df_locations = merge_locations.convert_json_to_columns(df)

#data only xmin around events
df_locations_events = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/xmin_around_events/locations_esm_timeperiod_5 min.csv_JSONconverted.pkl")

#load labels
sensors_included = "all"
with open("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_" + sensors_included + "_transformed_labeled_dict.pkl",
        'rb') as f:
    dict_label = pickle.load(f)
#
class Clustering:
    def __init__(self, df):
        self.df = df









# for one participant: get the locations for labels for before & after sleep for every participant

# add label column
label_column_name = "label_before and after sleep"
df_locations_events = labeling_sensor_df(df_locations_events, dict_label, label_column_name, ESM_identifier_column = "ESM_timestamp")
def get_locations_for_labels(df, label_column_name):
    # initialize lists
    participants = []
    dates = []
    times = []
    labels = []
    location_latitude = []
    location_longitude = []

    # delete rows in which label is nan
    number_of_events = len(df["ESM_timestamp"].unique())
    df = df.dropna(subset=[label_column_name])
    print("Number of Events which have been dropped because of missing label: {}".format(number_of_events - (len(df["ESM_timestamp"].unique()))))
    print("Number of Events which have been kept: {}".format(len(df["ESM_timestamp"].unique())))

    for participant in df.loc_device_id.unique():
        for event in df[df["loc_device_id"] == participant]["ESM_timestamp"].unique():
            # get the location just at the event
            df_event = df[(df["loc_device_id"] == participant) & (df["ESM_timestamp"] == event)]
            # get record with timestamp closest to ESM_timestamp
            df_event["timestamp"] = pd.to_datetime(df_event["timestamp"])
            df_event["ESM_timestamp"] = pd.to_datetime(df_event["ESM_timestamp"])
            df_event = df_event.iloc[(df_event["timestamp"] - df_event["ESM_timestamp"]).abs().argsort()[:1]]

            # add to lists
            participants.append(participant)
            dates.append(df_event["ESM_timestamp"].dt.date.values[0])
            times.append(df_event["ESM_timestamp"].dt.time.values[0])
            labels.append(df_event[label_column_name].values[0])
            location_latitude.append(df_event["loc_double_latitude"].values[0])
            location_longitude.append(df_event["loc_double_longitude"].values[0])

    # create dataframe
    df_locations_for_labels = pd.DataFrame({"participant": participants, "date": dates, "time": times, "label": labels, "latitude": location_latitude, "longitude": location_longitude})

    return df_locations_for_labels

df_locations_for_labels = get_locations_for_labels(df_locations_events, label_column_name)



Miranda
cf2dfa9b-596b-4a72-a4da-cb3777a31cc7  lying on the couch              11
                                      lying in bed after sleeping      6
                                      lying in bed at other times      6
                                      lying in bed before sleeping     5
Benedikt
8206b501-20e7-4851-8b79-601af9e4485f  lying in bed after sleeping      6
                                      lying in bed at other times      4
                                      lying in bed before sleeping     2
Lea
590f0faf-d932-4a57-998d-e3da667a91dc  lying on the couch               8
                                      lying in bed at other times      2
                                      lying in bed before sleeping     2
                                      lying in bed after sleeping      1
Bella
ad064def-9d72-44ff-96b2-f3b3d90d1079  lying in bed before sleeping     4
                                      lying in bed after sleeping      3
                                      lying in bed at other times      2
                                      lying on the couch               1


# find the most frequent location during every night of the participants: take for every night the location with the highest frequency
starthour = 0
endhour = 6
def most_frequent_location(df, starthour, endhour):
    # initialize lists
    participants = []
    dates = []
    starthour_list = []
    endhour_list = []
    most_frequent_location_latitude = []
    most_frequent_location_longitude = []
    location_counts_this_night = []

    for participant in df["loc_device_id"].unique():
        df_participant = df[df["loc_device_id"] == participant].copy()

        # filter for timeperiod
        ## check type of "loc_timestamp": if unix timestamp and convert to datetime
        if np.issubdtype(df_participant["loc_timestamp"].iloc[0], np.integer):
            # filter for timeperiod
            df_participant["loc_timestamp"] = pd.to_datetime(df_participant["loc_timestamp"], unit="ms")
        else:
            df_participant["loc_timestamp"] = pd.to_datetime(df_participant["loc_timestamp"])

        # filter for timeperiod
        df_participant = df_participant[df_participant["loc_timestamp"].dt.hour >= starthour]
        df_participant = df_participant[df_participant["loc_timestamp"].dt.hour < endhour]

        # iterate through different dates
        for date in df_participant["loc_timestamp"].dt.date.unique():
            df_participant_date = df_participant[df_participant["loc_timestamp"].dt.date == date].copy()

            #find most frequent location
            df_participant_date["location"] = df_participant_date["loc_double_latitude"].astype(str) + "_" + df["loc_double_longitude"].astype(str)

            # find location where user is most of the time
            most_frequent_location = df_participant_date["location"].value_counts().index[0]
            print("The number of locations of participant {} on {} is {}".format(participant, date, df_participant_date["location"].value_counts()))

            # save results
            participants.append(participant)
            dates.append(date)
            starthour_list.append(starthour)
            endhour_list.append(endhour)
            most_frequent_location_latitude.append(most_frequent_location.split("_")[0])
            most_frequent_location_longitude.append(most_frequent_location.split("_")[1])
            location_counts_this_night.append(df_participant_date["location"].value_counts())

    # create dataframe
    df_most_frequent_location = pd.DataFrame({"loc_device_id": participants,
                                                "date": dates,
                                                "starthour": starthour_list,
                                                "endhour": endhour_list,
                                                "most_frequent_location_latitude": most_frequent_location_latitude,
                                                "most_frequent_location_longitude": most_frequent_location_longitude,
                                                "location_counts_this_night": location_counts_this_night})

    return df_most_frequent_location

df_most_frequent_location = most_frequent_location(df, starthour, endhour)
#save most frequent location
df_most_frequent_location.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/locations/most_frequent_location.csv", index=False)

# see if I can find sleep locations of participants in the most frequent locations:
def calculate_distances_between_labels_and_most_frequent_locations(df_locations_for_labels, df_most_frequent_location):
    #iterate through df_locations_for_labels
    for index, row in df_locations_for_labels.iterrows():
        # calculate distance between event and most frequent location (for same partipant) for every record in most frequent locations
        df_most_frequent_location["distance"] = df_most_frequent_location.apply(lambda x: geopy.distance.geodesic((x["most_frequent_location_latitude"], x["most_frequent_location_longitude"]), (row["latitude"], row["longitude"])).m, axis=1)

        # find the row of df_most_frequent_location that which contains distance minimum (for same participant as in row of df_locations_for_labels)
        try:
            record = df_most_frequent_location[df_most_frequent_location["loc_device_id"] == row["participant"]].sort_values(by="distance").iloc[0]
        except:
            df_locations_for_labels.loc[index, "distance_to_most_frequent_location"] = np.nan
            df_locations_for_labels.loc[index, "most_frequent_location_latitude"] = np.nan
            df_locations_for_labels.loc[index, "most_frequent_location_longitude"] = np.nan
            df_locations_for_labels.loc[index, "most_frequent_location_date"] = np.nan
            df_locations_for_labels.loc[index, "most_frequent_location_starthour"] = np.nan
            df_locations_for_labels.loc[index, "most_frequent_location_endhour"] = np.nan
        else:
            record = df_most_frequent_location[df_most_frequent_location["loc_device_id"] == row["participant"]].sort_values(by="distance").iloc[0]
            df_locations_for_labels.loc[index, "distance_to_most_frequent_location"] = record["distance"]
            df_locations_for_labels.loc[index, "most_frequent_location_latitude"] = record["most_frequent_location_latitude"]
            df_locations_for_labels.loc[index, "most_frequent_location_longitude"] = record["most_frequent_location_longitude"]
            df_locations_for_labels.loc[index, "most_frequent_location_date"] = record["date"]
            df_locations_for_labels.loc[index, "most_frequent_location_starthour"] = record["starthour"]
            df_locations_for_labels.loc[index, "most_frequent_location_endhour"] = record["endhour"]

    return df_locations_for_labels

df_locations_for_labels = calculate_distances_between_labels_and_most_frequent_locations(df_locations_for_labels, df_most_frequent_location)


# find the most frequent location for each participant for specific timeperiod: use kmeans clustering & silhouette score to find optimal number of clusters
## based on https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

# compute for night
starthour = 0
endhour = 6


# Generating the sample data from make_blobs
# This particular setting has one distinct cluster and 3 clusters placed close
# together.
X, y = make_blobs(
    n_samples=500,
    n_features=2,
    centers=4,
    cluster_std=1,
    center_box=(-10.0, 10.0),
    shuffle=True,
    random_state=1,
)  # For reproducibility


#X = df[["loc_double_latitude", "loc_double_longitude"]].values

output_path = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/locations/"
def cluster_for_timeperiod(df, starthour, endhour):
    range_n_clusters = [2, 3, 4, 5, 6]

    # initialize lists
    participants = []
    number_clusters = []
    silhouette_scores = []
    number_entries = []
    cluster_1_latitude = []
    cluster_1_longitude = []
    cluster_1_entries_in_50m_range = []
    cluster_2_latitude = []
    cluster_2_longitude = []
    cluster_2_entries_in_50m_range = []
    cluster_3_latitude = []
    cluster_3_longitude = []
    cluster_3_entries_in_50m_range = []
    cluster_4_latitude = []
    cluster_4_longitude = []
    cluster_4_entries_in_50m_range = []
    cluster_5_latitude = []
    cluster_5_longitude = []
    cluster_5_entries_in_50m_range = []
    cluster_6_latitude = []
    cluster_6_longitude = []
    cluster_6_entries_in_50m_range = []
    cluster_7_latitude = []
    cluster_7_longitude = []
    cluster_7_entries_in_50m_range = []
    cluster_8_latitude = []
    cluster_8_longitude = []
    cluster_8_entries_in_50m_range = []
    cluster_9_latitude = []
    cluster_9_longitude = []
    cluster_9_entries_in_50m_range = []
    cluster_10_latitude = []
    cluster_10_longitude = []
    cluster_10_entries_in_50m_range = []
    df_summary = pd.DataFrame(columns= ["participant", "number_clusters", "silhouette_score", "number_entries", "cluster_1_latitude", "cluster_1_longitude", "cluster_1_entries_in_50m_range", "cluster_2_latitude", "cluster_2_longitude", "cluster_2_entries_in_50m_range", "cluster_3_latitude", "cluster_3_longitude", "cluster_3_entries_in_50m_range", "cluster_4_latitude", "cluster_4_longitude", "cluster_4_entries_in_50m_range", "cluster_5_latitude", "cluster_5_longitude", "cluster_5_entries_in_50m_range", "cluster_6_latitude", "cluster_6_longitude", "cluster_6_entries_in_50m_range", "cluster_7_latitude", "cluster_7_longitude", "cluster_7_entries_in_50m_range", "cluster_8_latitude", "cluster_8_longitude", "cluster_8_entries_in_50m_range", "cluster_9_latitude", "cluster_9_longitude", "cluster_9_entries_in_50m_range", "cluster_10_latitude", "cluster_10_longitude", "cluster_10_entries_in_50m_range"])

    counter = 1
    number_participants = len(df["loc_device_id"].unique())
    for participant in tqdm(df["loc_device_id"].unique()):
        print("Participant " + str(counter) + " of " + str(number_participants) + " started")
        # create dataframe for participant with only the timeperiod of interest
        df_participant = df[df["loc_device_id"] == participant].copy()

        # filter for timeperiod
        ## check type of "loc_timestamp": if unix timestamp and convert to datetime
        if np.issubdtype(df_participant["loc_timestamp"].iloc[0], np.integer):
            # filter for timeperiod
            df_participant["loc_timestamp"] = pd.to_datetime(df_participant["loc_timestamp"], unit="ms")
        else:
            df_participant["loc_timestamp"] = pd.to_datetime(df_participant["loc_timestamp"])
        print("timestamp converted to datetime")

        # filter for timeperiod
        df_participant = df_participant[df_participant["loc_timestamp"].dt.hour >= starthour]
        df_participant = df_participant[df_participant["loc_timestamp"].dt.hour < endhour]
        print("timeperiod filtered, datasize: " + str(len(df_participant)))

        # exception case 1: check if there are at least 2 entries in the timeperiod
        if len(df_participant) < 3600:
            print("participant", participant, "has less than 1 hour of GPS data (3600 entries) in the timeperiod. She/he will be excluded from the analysis.")
            counter += 1
            continue

        # exception case 2: if there is only one position for the participant, skip
        if len(df_participant["loc_double_latitude"].value_counts()) == 1:

            # summarize the important numbers for this participant
            # add np.nan row to df_summary
            df_summary.loc[len(df_summary)] = np.nan

            # add numbers for this participant into last row of df_summary
            df_summary.loc[len(df_summary) - 1, "participant"] = participant
            df_summary.loc[len(df_summary) - 1, "number_clusters"] = 1
            df_summary.loc[len(df_summary) - 1, "silhouette_score"] = 1
            df_summary.loc[len(df_summary) - 1, "number_entries"] = len(df_participant)
            df_summary.loc[len(df_summary) - 1, "cluster_1_latitude"] = df_participant["loc_double_latitude"].iloc[0]
            df_summary.loc[len(df_summary) - 1, "cluster_1_longitude"] = df_participant["loc_double_longitude"].iloc[0]
            df_summary.loc[len(df_summary) - 1, "cluster_1_entries_in_50m_range"] = len(df_participant)
            print("Participant " + participant + " had just one location")
            print("Participant " + str(counter) + " of " + str(number_participants) + " done.")
            counter += 1
            continue

        # create numpy array for clustering
        X = df_participant[["loc_double_latitude", "loc_double_longitude"]].values

        # find optimal number of clusters
        ## initialize variables
        number_clusters = []
        silhouette_averages = []
        for n_clusters in range_n_clusters:
            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(X)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(X, cluster_labels)
            print(
                "For participant",
                participant,
                "For n_clusters =",
                n_clusters,
                "The average silhouette_score is :",
                silhouette_avg,
            )

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(
                X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
            )

            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(
                centers[:, 0],
                centers[:, 1],
                marker="o",
                c="white",
                alpha=1,
                s=200,
                edgecolor="k",
            )

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(
                "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
                % n_clusters,
                fontsize=14,
                fontweight="bold",
            )

            plt.show()
            # save figure
            fig.savefig(
                os.path.join(
                    output_path,
                    "hours-{}-{}_silhouette_analysis_participant-{}_n_clusters-{}_silhouettescore-{}.png".format(
                         starthour, endhour, participant, n_clusters, silhouette_avg
                    ),
                )
            )


            #save relevant numbers
            number_clusters.append(n_clusters)
            silhouette_averages.append(silhouette_avg)

            # break if silhouette score is above 0.9
            if silhouette_avg > 0.9:
                print("Silhouette score is above 0.9 with cluster number: " + str(n_clusters) + " and participant " + participant + ", breaking loop")
                break

        # calculating clustering with optimal number of clusters
        optimal_number_clusters = number_clusters[np.argmax(silhouette_averages)]

        clusterer = KMeans(n_clusters=optimal_number_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)
        centers = clusterer.cluster_centers_
        # The silhouette_score
        silhouette_avg = silhouette_score(X, cluster_labels)

        # calculate distance for each point in cluster to center of cluster
        df_clusters = pd.DataFrame(X, columns=["loc_double_latitude", "loc_double_longitude"])
        df_clusters["cluster"] = cluster_labels
        df_clusters["distance_to_clustercenter"] = 0

        df_clusters_new = pd.DataFrame(columns=["loc_double_latitude", "loc_double_longitude", "cluster", "distance_to_clustercenter"])
        for cluster in range(len(centers)):
            df_clusters_currentcluster = df_clusters[df_clusters["cluster"] == cluster]
            df_clusters_currentcluster["distance_to_clustercenter"] = df_clusters_currentcluster.apply(
                lambda row: geopy.distance.geodesic(
                    (row["loc_double_latitude"], row["loc_double_longitude"]),
                    (centers[cluster][0], centers[cluster][1]),
                ).m,
                axis=1,
            )
            #concatenate
            df_clusters_new = pd.concat([df_clusters_new, df_clusters_currentcluster])

        # delete all entries with distance of point to cluster centroid > 50m
        print("Number of entries before deleting points which are not in 50 meter range of cluster centroid: ", len(df_clusters_new))
        df_clusters_new = df_clusters_new[df_clusters_new["distance_to_clustercenter"] < 50]
        print("Number of entries after deleting points which are not in 50 meter range of cluster centroid: ", len(df_clusters_new))

        # summarize the important numbers for this participant
        #add np.nan row to df_summary
        df_summary.loc[len(df_summary)] = np.nan

        # add numbers for this participant into last row of df_summary
        df_summary.loc[len(df_summary)-1, "participant"] = participant
        df_summary.loc[len(df_summary)-1, "number_clusters"] = optimal_number_clusters
        df_summary.loc[len(df_summary)-1, "silhouette_score"] = silhouette_avg
        df_summary.loc[len(df_summary)-1, "number_entries"] = len(df_clusters_new)
        df_summary.loc[len(df_summary) - 1, "cluster_1_latitude"] = centers[0][0]
        df_summary.loc[len(df_summary) - 1, "cluster_1_longitude"] = centers[0][1]
        df_summary.loc[len(df_summary) - 1, "cluster_1_entries_in_50m_range"] = len(df_clusters_new[df_clusters_new["cluster"] == 0])
        if optimal_number_clusters > 1:
            df_summary.loc[len(df_summary) - 1, "cluster_2_latitude"] = centers[1][0]
            df_summary.loc[len(df_summary) - 1, "cluster_2_longitude"] = centers[1][1]
            df_summary.loc[len(df_summary) - 1, "cluster_2_entries_in_50m_range"] = len(df_clusters_new[df_clusters_new["cluster"] == 1])
        if optimal_number_clusters > 2:
            df_summary.loc[len(df_summary) - 1, "cluster_3_latitude"] = centers[2][0]
            df_summary.loc[len(df_summary) - 1, "cluster_3_longitude"] = centers[2][1]
            df_summary.loc[len(df_summary) - 1, "cluster_3_entries_in_50m_range"] = len(df_clusters_new[df_clusters_new["cluster"] == 2])
        if optimal_number_clusters > 3:
            df_summary.loc[len(df_summary) - 1, "cluster_4_latitude"] = centers[3][0]
            df_summary.loc[len(df_summary) - 1, "cluster_4_longitude"] = centers[3][1]
            df_summary.loc[len(df_summary) - 1, "cluster_4_entries_in_50m_range"] = len(df_clusters_new[df_clusters_new["cluster"] == 3])
        if optimal_number_clusters > 4:
            df_summary.loc[len(df_summary) - 1, "cluster_5_latitude"] = centers[4][0]
            df_summary.loc[len(df_summary) - 1, "cluster_5_longitude"] = centers[4][1]
            df_summary.loc[len(df_summary) - 1, "cluster_5_entries_in_50m_range"] = len(df_clusters_new[df_clusters_new["cluster"] == 4])
        if optimal_number_clusters > 5:
            df_summary.loc[len(df_summary) - 1, "cluster_6_latitude"] = centers[5][0]
            df_summary.loc[len(df_summary) - 1, "cluster_6_longitude"] = centers[5][1]
            df_summary.loc[len(df_summary) - 1, "cluster_6_entries_in_50m_range"] = len(df_clusters_new[df_clusters_new["cluster"] == 5])
        if optimal_number_clusters > 6:
            df_summary.loc[len(df_summary) - 1, "cluster_7_latitude"] = centers[6][0]
            df_summary.loc[len(df_summary) - 1, "cluster_7_longitude"] = centers[6][1]
            df_summary.loc[len(df_summary) - 1, "cluster_7_entries_in_50m_range"] = len(df_clusters_new[df_clusters_new["cluster"] == 6])
        if optimal_number_clusters > 7:
            df_summary.loc[len(df_summary) - 1, "cluster_8_latitude"] = centers[7][0]
            df_summary.loc[len(df_summary) - 1, "cluster_8_longitude"] = centers[7][1]
            df_summary.loc[len(df_summary) - 1, "cluster_8_entries_in_50m_range"] = len(df_clusters_new[df_clusters_new["cluster"] == 7])
        if optimal_number_clusters > 8:
            df_summary.loc[len(df_summary) - 1, "cluster_9_latitude"] = centers[8][0]
            df_summary.loc[len(df_summary) - 1, "cluster_9_longitude"] = centers[8][1]
            df_summary.loc[len(df_summary) - 1, "cluster_9_entries_in_50m_range"] = len(df_clusters_new[df_clusters_new["cluster"] == 8])
        if optimal_number_clusters > 9:
            df_summary.loc[len(df_summary) - 1, "cluster_10_latitude"] = centers[9][0]
            df_summary.loc[len(df_summary) - 1, "cluster_10_longitude"] = centers[9][1]
            df_summary.loc[len(df_summary) - 1, "cluster_10_entries_in_50m_range"] = len(df_clusters_new[df_clusters_new["cluster"] == 9])

        #print progress
        print("Done with participant " + str(counter) + " of " + str(number_participants) + " participants.")
        counter += 1

    return df_summary


# compute nights and daysin chunks
#night
starthour = 0
endhour = 6
#day
starthour = 9
endhour = 18
chunk_counter = 1
for df in pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/locations_all_sorted.csv", chunksize=500000):
    time_start = time.time()
    df_summary = cluster_for_timeperiod(df, starthour, endhour)
    df_summary.to_csv(
        os.path.join(
            output_path,
            "hours-{}-{}_freqquent_locations_summary_chunk-{}.csv".format(
                starthour, endhour, chunk_counter
            ),
        ), index=False)
    chunk_counter += 1
    time_end = time.time()
    print("Time needed for chunk: " + str((time_end - time_start)/60) + " minutes")
    print("Done with chunk " + str(chunk_counter) + " chunks.")



# iterate through chunks and merge them
#night
starthour = 0
endhour = 6
chunk_number = 5
#day
starthour = 9
endhour = 18
chunk_number = 5

folder = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/locations/"
for i in range(1, chunk_number + 1):
    df_chunk = pd.read_csv(folder + "hours-"+ str(starthour) +"-" + str(endhour) + "_freqquent_locations_summary_chunk-" + str(i) + ".csv")
    if i == 1:
        df_merged = df_chunk
    else:
        df_merged = pd.concat([df_merged, df_chunk], ignore_index=True)

df_merged.to_csv(folder + "hours-"+ str(starthour) +"-" + str(endhour) + "_freqquent_locations_summary_merged.csv", index=False)

#TODO merge locations that are close to each other


#TODO delete in day and night most frequent locations the ones which dont represent many values
#load data
starthour = 0
endhour = 6
df = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/locations/hours-"+ str(starthour)+"-"+ str(endhour) +"_freqquent_locations_summary_merged.csv")

# delete locations which are not frequent enough
# if %values < (number_of_entries/number_clusters+1) then delete
for index, row in df.iterrows():
    for i in range(0, int(row["number_clusters"])):
        if row["cluster_" + str(i+1) + "_entries_in_50m_range"] < (row["number_entries"]/(row["number_clusters"]+1)):
            df.loc[index, "cluster_" + str(i+1) + "_latitude"] = np.nan
            df.loc[index, "cluster_" + str(i+1) + "_longitude"] = np.nan
            df.loc[index, "cluster_" + str(i+1) + "_entries_in_50m_range"] = np.nan
df.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/locations/hours-"+ str(starthour) +"-"+ str(endhour) +"_freqquent_locations_summary_merged.csv", index=False)
# label all 


#testing
coords_1 = (51.332278828745224, 12.397581165608605)
coords_2 = (51.33217582765464, 12.397809464756133)

print(geopy.distance.geodesic(coords_1, coords_2).m)


