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

# for clustering
import haversine
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist

# for GPS feature creation
from datetime import timedelta
import pytz




#endregion


#region load data
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
#endregion


#testaread
# print value counts in chunks for every device_id
print("start printing value counts")
for device_id in df_locations["device_id"].unique():
    print("participant " + str(device_id))
    print(df_locations[df_locations["device_id"] == device_id]["chunk"].value_counts())
    # print how many unique pairs of latitude and longitude there are
    print("number of unique pairs of latitude and longitude: " + str(len(df_locations[df_locations["device_id"] == device_id][["latitude", "longitude"]].drop_duplicates())))

#region find the most frequent location for each participant for specific timeperiod: use kmeans clustering & silhouette score to find optimal number of clusters
## based on https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
class GPS_clustering:
    # create haversine distance matrix based on latitude and longitude of locations
    def distance_matrix_haversine(latitude, longitude):
        if type(latitude) != np.ndarray:
            latitude = np.array(latitude)
        if type(longitude) != np.ndarray:
            longitude = np.array(longitude)
        distance_matrix = np.zeros((len(latitude), len(latitude)))
        for i in range(len(latitude)):
            for j in range(len(latitude)):
                distance_matrix[i, j] = haversine.haversine((latitude[i], longitude[i]), (latitude[j], longitude[j]))
        return distance_matrix

    def agg_clustering_computing_centroids(df_locations, distance_threshold):

        # drop columns "cluster_label", "cluster_latitude_mean", "cluster_longitude_mean" in case they already exist
        if "cluster_label" in df_locations.columns:
            df_locations = df_locations.drop(columns=["cluster_label", "cluster_latitude_mean", "cluster_longitude_mean"])

        # to make code more efficient: drop duplicate GPS values in beginning and add them again after clustering is done
        print("start dropping duplicates. Size of dataframe: " + str(len(df_locations)) + " rows")
        df_locations_including_duplicates = df_locations.copy()
        df_locations = df_locations.drop_duplicates(subset=["latitude", "longitude"], keep="first")
        print("duplicate dropping reduced dataframe from " + str(len(df_locations_including_duplicates)) + " to " + str(len(df_locations)) + " rows")

        ## create distance matrix
        print("start computing distance matrix. Size of dataframe: " + str(len(df_locations)) + " rows")
        X = np.array(list(zip(df_locations["latitude"], df_locations["longitude"])))
        distance_matrix = cdist(X, X, metric=haversine.haversine)
        #distance_matrix = GPS_clustering.distance_matrix_haversine(df_locations["latitude"], df_locations["longitude"])

        ## cluster locations
        print("start clustering")
        agg_clustering = AgglomerativeClustering(linkage="complete", n_clusters=None,
                                                 distance_threshold=distance_threshold,
                                                 metric="precomputed").fit(distance_matrix)
        print("clustering finished")

        # add cluster labels to dataframe
        df_locations["cluster_label"] = agg_clustering.labels_

        # add duplicate GPS values again
        # add df_locations_including_duplicates the cluster labels of the rows with the same GPS values
        print("start adding duplicates back to dataframe. Size of dataframe: " + str(len(df_locations)) + " rows")
        df_merged = df_locations_including_duplicates.merge(df_locations[["latitude", "longitude", "cluster_label"]], on=["latitude", "longitude"], how="left")
        # set as index of df_merged the index of df_locations_including_duplicates
        df_merged = df_merged.set_index(df_locations_including_duplicates.index)
        df_locations_including_duplicates['cluster_label'] = df_merged['cluster_label']
        df_locations = pd.concat([df_locations, df_locations_including_duplicates], axis=0)
        # delete any rows which have same index (since otherwise for each record which had duplicate GPS values, it is now really duplicated)
        df_locations = df_locations[~df_locations.index.duplicated(keep='first')]
        print("adding duplicates back to dataframe finished. Size of dataframe: " + str(len(df_locations)) + " rows")

        # compute centroids of clusters
        print("start computing centroid")
        df_locations["cluster_latitude_mean"] = df_locations.groupby("cluster_label")["latitude"].transform("mean")
        df_locations["cluster_longitude_mean"] = df_locations.groupby("cluster_label")["longitude"].transform("mean")

        # create analytics dataframe
        df_analytics = pd.DataFrame(columns=["participant", "cluster_label", "number_points", "latitude_mean", "longitude_mean"])
        for cluster_label in df_locations["cluster_label"].unique():
            print("started with cluster label " + str(cluster_label) + " of " + str(len(df_locations["cluster_label"].unique())))
            df_cluster = df_locations[df_locations["cluster_label"] == cluster_label]
            #concatenate to analytics dataframe
            df_analytics.loc[len(df_analytics)] = [df_cluster["device_id"].unique()[0], cluster_label, len(df_cluster),
                                                   df_cluster["cluster_latitude_mean"].iloc[0],
                                                   df_cluster["cluster_longitude_mean"].iloc[0]]

        return df_locations, df_analytics

    # for participants for which the clusters have been computed for chunks of their data: merge clusters of the chunks

    def merge_clusters_of_chunks(df_locations, participant_list):
        df_analytics = pd.DataFrame(
            columns=["participant", "chunk", "cluster_base", "cluster_to_merge", "has_this_cluster_been_merged?"])

        for participant in participant_list:
            print("start with participant " + str(participant) + "...")
            df_participant = df_locations[df_locations["device_id"] == participant]

            # TEMPORARY: delete records with NaN in "chunk"
            #df_participant = df_participant[df_participant["chunk"].notna()]

            # create dataframe which contains chunks, cluster_label, cluster_latitude_mean, cluster_longitude_mean, and number of records
            df_chunk_clusters_GPS = pd.DataFrame(df_participant.groupby(["chunk", "cluster_label"]).size())
            df_chunk_clusters_GPS.columns = ["number_of_records"]
            df_chunk_clusters_GPS["cluster_latitude_mean"] = df_participant.groupby(["chunk", "cluster_label"])[
                "cluster_latitude_mean"].mean()
            df_chunk_clusters_GPS["cluster_longitude_mean"] = df_participant.groupby(["chunk", "cluster_label"])[
                "cluster_longitude_mean"].mean()
            # seperate now the index: chunks and cluster_labels should be each in one column
            df_chunk_clusters_GPS = df_chunk_clusters_GPS.reset_index()

            for chunk in df_chunk_clusters_GPS["chunk"].unique():
                print("start with chunk " + str(chunk) + "...")
                dict_clusters_less_than_100m_apart = {}
                df_chunk_clusters_GPS_chunk = df_chunk_clusters_GPS[df_chunk_clusters_GPS["chunk"] == chunk]
                df_chunk_clusters_GPS_notchunk = df_chunk_clusters_GPS[df_chunk_clusters_GPS["chunk"] != chunk]
                # reset index
                df_chunk_clusters_GPS_chunk = df_chunk_clusters_GPS_chunk.reset_index()
                df_chunk_clusters_GPS_notchunk = df_chunk_clusters_GPS_notchunk.reset_index()
                # create distance matrix between clusters in this chunk and clusters in other chunks

                distance_matrix = cdist(
                    df_chunk_clusters_GPS_chunk[["cluster_latitude_mean", "cluster_longitude_mean"]],
                    df_chunk_clusters_GPS_notchunk[["cluster_latitude_mean", "cluster_longitude_mean"]],
                    metric=haversine.haversine)

                #  for every cluster label in chunk: check if there is a cluster in other chunks which is closer than 100m
                row_counter = 0
                for row in distance_matrix:
                    # get indices of clusters in other chunks which are closer than 100m
                    indices = np.where(row < 0.1)[0]
                    # add the chunks and cluster labels of these clusters as list to dictionary with key = cluster label in chunk
                    chunks = df_chunk_clusters_GPS_notchunk.loc[indices, "chunk"].tolist()
                    cluster_labels = df_chunk_clusters_GPS_notchunk.loc[indices, "cluster_label"].tolist()
                    if len(chunks) > 0:
                        dict_clusters_less_than_100m_apart[df_chunk_clusters_GPS_chunk.loc[row_counter, "cluster_label"]] = \
                            list(zip(chunks, cluster_labels))

                    row_counter += 1

                # check for each close clusters, if the maximal distance of any two points is less than 150 meters
                for cluster_label in dict_clusters_less_than_100m_apart.keys():
                    print("start with cluster label " + str(cluster_label) + " of " + str(
                        len(dict_clusters_less_than_100m_apart.keys())))

                    # get all records of this cluster in this chunk
                    df_cluster = df_participant[(df_participant["chunk"] == chunk) & (
                                df_participant["cluster_label"] == cluster_label)]

                    for close_cluster in dict_clusters_less_than_100m_apart[cluster_label]:
                        print("start with close cluster " + str(close_cluster) + " of " + str(
                            len(dict_clusters_less_than_100m_apart[cluster_label])))
                        close_cluster_chunk = close_cluster[0]
                        close_cluster_label = close_cluster[1]

                        # get all records of close cluster in other chunk
                        df_close_cluster = df_participant[(df_participant["chunk"] == close_cluster_chunk) & (
                                    df_participant["cluster_label"] == close_cluster_label)]

                        # drop in both clusters any duplicates in latitude and longitude
                        df_cluster = df_cluster.drop_duplicates(subset=["latitude", "longitude"])
                        df_close_cluster = df_close_cluster.drop_duplicates(subset=["latitude", "longitude"])

                        # calculate distance matrix between all records of this cluster and all records of close cluster
                        distance_matrix = cdist(df_cluster[["latitude", "longitude"]],
                                                df_close_cluster[["latitude", "longitude"]], metric=haversine.haversine)

                        # check if there is a distance greater than 150m
                        if np.max(distance_matrix) > 0.15:
                            # if yes: delete the close cluster from the dictionary
                            dict_clusters_less_than_100m_apart[cluster_label].remove(close_cluster)

                            # add analytics data zu df_analytics using loc
                            df_analytics.loc[len(df_analytics)] = [participant, chunk, cluster_label,
                                                                   close_cluster_label, "no"]
                            continue

                        else:
                            # if no: get the indices of the records of the close cluster in the df_locations dataframe
                            indices = df_locations[(df_locations["device_id"] == participant) & (
                                        df_locations["chunk"] == close_cluster_chunk) & (
                                                              df_locations["cluster_label"] == close_cluster_label)].index

                            # change the chunk and cluster_label of those records to the chunk and cluster_label of the cluster
                            df_locations.loc[indices, "chunk"] = chunk
                            df_locations.loc[indices, "cluster_label"] = cluster_label

                            # create new cluster_latitude_mean and cluster_longitude_mean for this new cluster for the same participant and chunk
                            df_locations.loc[(df_locations["device_id"] == participant) & (
                                        df_locations["chunk"] == chunk) & (
                                                            df_locations["cluster_label"] == cluster_label),
                            "cluster_latitude_mean"] = df_locations.loc[(df_locations["device_id"] == participant) & (
                                        df_locations["chunk"] == chunk) & (
                                                            df_locations["cluster_label"] == cluster_label),
                            "latitude"].mean()
                            df_locations.loc[(df_locations["device_id"] == participant) & (
                                        df_locations["chunk"] == chunk) & (
                                                            df_locations["cluster_label"] == cluster_label),
                            "cluster_longitude_mean"] = df_locations.loc[(df_locations["device_id"] == participant) & (
                                        df_locations["chunk"] == chunk) & (
                                                            df_locations["cluster_label"] == cluster_label),
                            "longitude"].mean()

                            # delete this record (identify by chunk and cluster_label) from df_chunk_clusters_GPS using drop
                            ## Reason: this cluster is now part of another cluster and therefore should not be considered in the next iteration
                            df_chunk_clusters_GPS = df_chunk_clusters_GPS.drop(
                                df_chunk_clusters_GPS[(df_chunk_clusters_GPS["chunk"] == close_cluster_chunk) &
                                                      (df_chunk_clusters_GPS[
                                                           "cluster_label"] == close_cluster_label)].index)

                            # add analytics data zu df_analytics using loc
                            df_analytics.loc[len(df_analytics)] = [participant, chunk, cluster_label,
                                                                   close_cluster_label, "yes"]

        return df_locations, df_analytics






    ### all functions below are created in the first iteration of classificaiton and might be outdated

    # create location clusters for the given timeperiod (i.e. day and night)
    def cluster_for_timeperiod(df, starthour, endhour, range_n_clusters, output_path):

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

        df_summary = pd.DataFrame(
            columns=["participant", "number_clusters", "silhouette_score", "number_entries", "cluster_1_latitude",
                     "cluster_1_longitude", "cluster_1_entries_in_50m_range", "cluster_2_latitude",
                     "cluster_2_longitude", "cluster_2_entries_in_50m_range", "cluster_3_latitude",
                     "cluster_3_longitude", "cluster_3_entries_in_50m_range", "cluster_4_latitude",
                     "cluster_4_longitude", "cluster_4_entries_in_50m_range", "cluster_5_latitude",
                     "cluster_5_longitude", "cluster_5_entries_in_50m_range", "cluster_6_latitude",
                     "cluster_6_longitude", "cluster_6_entries_in_50m_range", "cluster_7_latitude",
                     "cluster_7_longitude", "cluster_7_entries_in_50m_range", "cluster_8_latitude",
                     "cluster_8_longitude", "cluster_8_entries_in_50m_range", "cluster_9_latitude",
                     "cluster_9_longitude", "cluster_9_entries_in_50m_range", "cluster_10_latitude",
                     "cluster_10_longitude", "cluster_10_entries_in_50m_range"])

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
                print("participant", participant,
                      "has less than 1 hour of GPS data (3600 entries) in the timeperiod. She/he will be excluded from the analysis.")
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
                df_summary.loc[len(df_summary) - 1, "cluster_1_latitude"] = df_participant["loc_double_latitude"].iloc[
                    0]
                df_summary.loc[len(df_summary) - 1, "cluster_1_longitude"] = \
                df_participant["loc_double_longitude"].iloc[0]
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

                # save relevant numbers
                number_clusters.append(n_clusters)
                silhouette_averages.append(silhouette_avg)

                # break if silhouette score is above 0.9
                if silhouette_avg > 0.9:
                    print("Silhouette score is above 0.9 with cluster number: ",
                        n_clusters, " and participant ", participant, ", breaking loop")
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

            df_clusters_new = pd.DataFrame(
                columns=["loc_double_latitude", "loc_double_longitude", "cluster", "distance_to_clustercenter"])
            for cluster in range(len(centers)):
                df_clusters_currentcluster = df_clusters[df_clusters["cluster"] == cluster]
                df_clusters_currentcluster["distance_to_clustercenter"] = df_clusters_currentcluster.apply(
                    lambda row: geopy.distance.geodesic(
                        (row["loc_double_latitude"], row["loc_double_longitude"]),
                        (centers[cluster][0], centers[cluster][1]),
                    ).m,
                    axis=1,
                )
                # concatenate
                df_clusters_new = pd.concat([df_clusters_new, df_clusters_currentcluster])

            # delete all entries with distance of point to cluster centroid > 50m
            print("Number of entries before deleting points which are not in 50 meter range of cluster centroid: ",
                  len(df_clusters_new))
            df_clusters_new = df_clusters_new[df_clusters_new["distance_to_clustercenter"] < 50]
            print("Number of entries after deleting points which are not in 50 meter range of cluster centroid: ",
                  len(df_clusters_new))

            # summarize the important numbers for this participant
            # add np.nan row to df_summary
            df_summary.loc[len(df_summary)] = np.nan

            # add numbers for this participant into last row of df_summary
            df_summary.loc[len(df_summary) - 1, "participant"] = participant
            df_summary.loc[len(df_summary) - 1, "number_clusters"] = optimal_number_clusters
            df_summary.loc[len(df_summary) - 1, "silhouette_score"] = silhouette_avg
            df_summary.loc[len(df_summary) - 1, "number_entries"] = len(df_clusters_new)
            df_summary.loc[len(df_summary) - 1, "cluster_1_latitude"] = centers[0][0]
            df_summary.loc[len(df_summary) - 1, "cluster_1_longitude"] = centers[0][1]
            df_summary.loc[len(df_summary) - 1, "cluster_1_entries_in_50m_range"] = len(
                df_clusters_new[df_clusters_new["cluster"] == 0])
            if optimal_number_clusters > 1:
                df_summary.loc[len(df_summary) - 1, "cluster_2_latitude"] = centers[1][0]
                df_summary.loc[len(df_summary) - 1, "cluster_2_longitude"] = centers[1][1]
                df_summary.loc[len(df_summary) - 1, "cluster_2_entries_in_50m_range"] = len(
                    df_clusters_new[df_clusters_new["cluster"] == 1])
            if optimal_number_clusters > 2:
                df_summary.loc[len(df_summary) - 1, "cluster_3_latitude"] = centers[2][0]
                df_summary.loc[len(df_summary) - 1, "cluster_3_longitude"] = centers[2][1]
                df_summary.loc[len(df_summary) - 1, "cluster_3_entries_in_50m_range"] = len(
                    df_clusters_new[df_clusters_new["cluster"] == 2])
            if optimal_number_clusters > 3:
                df_summary.loc[len(df_summary) - 1, "cluster_4_latitude"] = centers[3][0]
                df_summary.loc[len(df_summary) - 1, "cluster_4_longitude"] = centers[3][1]
                df_summary.loc[len(df_summary) - 1, "cluster_4_entries_in_50m_range"] = len(
                    df_clusters_new[df_clusters_new["cluster"] == 3])
            if optimal_number_clusters > 4:
                df_summary.loc[len(df_summary) - 1, "cluster_5_latitude"] = centers[4][0]
                df_summary.loc[len(df_summary) - 1, "cluster_5_longitude"] = centers[4][1]
                df_summary.loc[len(df_summary) - 1, "cluster_5_entries_in_50m_range"] = len(
                    df_clusters_new[df_clusters_new["cluster"] == 4])
            if optimal_number_clusters > 5:
                df_summary.loc[len(df_summary) - 1, "cluster_6_latitude"] = centers[5][0]
                df_summary.loc[len(df_summary) - 1, "cluster_6_longitude"] = centers[5][1]
                df_summary.loc[len(df_summary) - 1, "cluster_6_entries_in_50m_range"] = len(
                    df_clusters_new[df_clusters_new["cluster"] == 5])
            if optimal_number_clusters > 6:
                df_summary.loc[len(df_summary) - 1, "cluster_7_latitude"] = centers[6][0]
                df_summary.loc[len(df_summary) - 1, "cluster_7_longitude"] = centers[6][1]
                df_summary.loc[len(df_summary) - 1, "cluster_7_entries_in_50m_range"] = len(
                    df_clusters_new[df_clusters_new["cluster"] == 6])
            if optimal_number_clusters > 7:
                df_summary.loc[len(df_summary) - 1, "cluster_8_latitude"] = centers[7][0]
                df_summary.loc[len(df_summary) - 1, "cluster_8_longitude"] = centers[7][1]
                df_summary.loc[len(df_summary) - 1, "cluster_8_entries_in_50m_range"] = len(
                    df_clusters_new[df_clusters_new["cluster"] == 7])
            if optimal_number_clusters > 8:
                df_summary.loc[len(df_summary) - 1, "cluster_9_latitude"] = centers[8][0]
                df_summary.loc[len(df_summary) - 1, "cluster_9_longitude"] = centers[8][1]
                df_summary.loc[len(df_summary) - 1, "cluster_9_entries_in_50m_range"] = len(
                    df_clusters_new[df_clusters_new["cluster"] == 8])
            if optimal_number_clusters > 9:
                df_summary.loc[len(df_summary) - 1, "cluster_10_latitude"] = centers[9][0]
                df_summary.loc[len(df_summary) - 1, "cluster_10_longitude"] = centers[9][1]
                df_summary.loc[len(df_summary) - 1, "cluster_10_entries_in_50m_range"] = len(
                    df_clusters_new[df_clusters_new["cluster"] == 9])

            # print progress
            print("Done with participant " + str(counter) + " of " + str(number_participants) + " participants.")
            counter += 1

        return df_summary

    # merge the clusters which are less than "threshold_distance" meters apart
    def merge_close_locations(df, threshold_distance):
        df_final = pd.DataFrame(
            columns=["participant", "cluster_latitude", "cluster_longitude", "cluster_entries_in_50m_range",
                     "number of entries"])

        # for every participant create dataframe containing centroids of all clusters
        for participant in df["participant"].unique():
            df_participant = df[df["participant"] == participant]
            for index, row in df_participant.iterrows():
                for i in range(1, int(row["number_clusters"]) + 1):
                    # add to df_final
                    df_final.loc[len(df_final)] = [participant, row["cluster_" + str(i) + "_latitude"],
                                                   row["cluster_" + str(i) + "_longitude"],
                                                   row["cluster_" + str(i) + "_entries_in_50m_range"],
                                                   (row["number_entries"] / row["number_clusters"])]

        # for every cluster centroid check if there is another cluster centroid closer than <<threshold>> meters
        dropped_indices = []
        for participant in df_final["participant"].unique():
            df_participant = df_final[df_final["participant"] == participant]
            for index, row in df_participant.iterrows():
                # check if index was dropped already. If yes, continue
                if index in dropped_indices:
                    continue
                # calculate if any other cluster centroid is closer than 50m
                for index2, row2 in df_participant.iterrows():
                    # check if index was dropped already. If yes, continue
                    if index2 in dropped_indices:
                        continue
                    if index != index2:
                        distance = geopy.distance.geodesic(
                            (row["cluster_latitude"], row["cluster_longitude"]),
                            (row2["cluster_latitude"], row2["cluster_longitude"])).m
                        print("Distance for participant " + str(participant) + " between cluster " + str(
                            index) + " and cluster " + str(index2) + " is " + str(distance))

                        if distance < threshold_distance:
                            # merge
                            df_final.loc[index, "cluster_latitude"] = (row["cluster_latitude"] + row2[
                                "cluster_latitude"]) / 2
                            df_final.loc[index, "cluster_longitude"] = (row["cluster_longitude"] + row2[
                                "cluster_longitude"]) / 2
                            df_final.loc[index, "cluster_entries_in_50m_range"] = row["cluster_entries_in_50m_range"] + \
                                                                                  row2["cluster_entries_in_50m_range"]
                            df_final.loc[index, "number of entries"] = df_final.loc[index, "number of entries"] + row2[
                                "number of entries"]
                            print("number of entries added: " + str(row2["number of entries"]))
                            # delete row2
                            print("number of entries dropped: " + str(df_final.loc[index2, "number of entries"]))
                            df_final = df_final.drop(index2)
                            print("Dropped index " + str(index2))
                            # add dropped index to list of dropped indices
                            dropped_indices.append(index2)

        # TODO if yes, merge the two clusters

        return df_final

    # delete locations which are not frequent enough
    def delete_locations_not_frequent_enough(df):
        df["percentage_values_in_50m_range"] = ""
        for participant in df["participant"].unique():
            df_participant = df[df["participant"] == participant]
            num_entries_participant = df_participant["number of entries"].sum()
            num_clusters_participant = len(df_participant)
            # sort df_participant by number of entries in 50m range (lowest first);
            # important since then the small clusters are deleted first
            df_participant = df_participant.sort_values(by=["cluster_entries_in_50m_range"], ascending=True)

            # iterate through clusters
            for index, row in df_participant.iterrows():
                # calculate percentage of values which are in 50m range around cluster centroid
                df.loc[index, "percentage_values_in_50m_range"] = row[
                                                                      "cluster_entries_in_50m_range"] / num_entries_participant
                # check if there is only one cluster left for participant; if yes, continue
                if len(df[df["participant"] == participant]) == 1:
                    continue
                # check if %values < (number_of_entries/number_clusters+1)
                if (row["cluster_entries_in_50m_range"] < (num_entries_participant / num_clusters_participant + 1)):
                    df = df.drop(index)

            df_participant["percentage_values_in_50m_range"] = df_participant["cluster_entries_in_50m_range"]/df_participant["number of entries"]

        return df
        #TODO something wrong with the calculation of "number of entries" here since there are more points around the cluster then in total

    # identify home, work, other sleep locations, other frequent locations
    def classify_locations(df_frequentlocations_day, df_frequentlocations_night, threshold_distance):
        # identify home location & sleep locations
        ## if there is only one location, it is the home location
        ## if there are more then one location, the one with the most entries is the home location and the
        ## other ones are "other sleep locations"
        locations_predictions = dict()
        for participant in df_frequentlocations_night["participant"].unique():
            df_participant = df_frequentlocations_night[df_frequentlocations_night["participant"] == participant]
            # initialize lists
            location_name = []
            latitude = []
            longitude = []
            # if there is only one location for participant, it is the home location
            if len(df_participant) == 1:
                # add this location to a list of locations
                location_name.append("home")
                latitude.append(df_participant["cluster_latitude"].iloc[0])
                longitude.append(df_participant["cluster_longitude"].iloc[0])

            # if there are more than one location for participant, set the location with the highest percentage of values in 50m range as home location and the other locations as "other sleep place"
            else:
                # sort df_participant by percentage of values in 50m range (highest first)
                df_participant = df_participant.sort_values(by=["percentage_values_in_50m_range"], ascending=False)
                # add the location with the highest percentage of values in 50m range to a list of locations
                location_name.append("home")
                latitude.append(df_participant.iloc[0]["cluster_latitude"])
                longitude.append(df_participant.iloc[0]["cluster_longitude"])
                # add the other locations to a list of locations
                for index, row in df_participant.iloc[1:].iterrows():
                    location_name.append("other sleep place")
                    latitude.append(row["cluster_latitude"])
                    longitude.append(row["cluster_longitude"])
            # add the list of locations to a dictionary
            locations_predictions[participant] = pd.DataFrame(
                {"location_name": location_name, "latitude": latitude, "longitude": longitude})

        # classify day-locations with locations from night-locations
        for participant in df_frequentlocations_day["participant"].unique():
            df_participant = df_frequentlocations_day[df_frequentlocations_day["participant"] == participant]
            # get the locations from the night-locations
            if participant in locations_predictions.keys():
                df_locations_night = locations_predictions[participant]
                # iterate through day-locations
                for index, row in df_participant.iterrows():
                    # iterate through night-locations
                    for index2, row2 in df_locations_night.iterrows():
                        # calculate distance between day-location and night-location
                        distance = geopy.distance.distance(
                            (row["cluster_latitude"], row["cluster_longitude"]),
                            (row2["latitude"], row2["longitude"])).m
                        # if distance < threshold_distance, add the location name from the night-locations to the day-locations
                        if distance < threshold_distance:
                            df_frequentlocations_day.loc[index, "location_name"] = row2["location_name"]
            else:
                print("No night-locations for participant " + participant)
        # identify work locations & other frequent locations
        ## if there is only one location apart from night locations, it is the work location
        ## if there are more than one location apart from night locations, set the location with the highest percentage
        ## of values in 50m range as work location and the other locations as "other frequent place"
        for participant in df_frequentlocations_day["participant"].unique():
            df_participant = df_frequentlocations_day[df_frequentlocations_day["participant"] == participant]
            # initialize lists
            location_name = []
            latitude = []
            longitude = []
            # check if there is only one location for participant without a location name
            if len(df_participant[df_participant["location_name"] == ""]) == 1:
                # add this location to a list of locations
                location_name.append("work")
                latitude.append(df_participant["cluster_latitude"].iloc[0])
                longitude.append(df_participant["cluster_longitude"].iloc[0])
            # if there are more than one location for participant without a location name, set the location with the highest percentage of values in 50m range as work location and the other locations as "other frequent place"
            else:
                # sort df_participant by percentage of values in 50m range (highest first)
                df_participant = df_participant.sort_values(by=["percentage_values_in_50m_range"], ascending=False)
                # add the location with the highest percentage of values in 50m range to a list of locations
                location_name.append("work")
                latitude.append(df_participant.iloc[0]["cluster_latitude"])
                longitude.append(df_participant.iloc[0]["cluster_longitude"])
                # add the other locations to a list of locations
                for index, row in df_participant.iloc[1:].iterrows():
                    location_name.append("other frequent place")
                    latitude.append(row["cluster_latitude"])
                    longitude.append(row["cluster_longitude"])
            # add the list of locations to a dictionary (concatenate to the locations from the night-locations if there are any)
            if participant in locations_predictions.keys():
                locations_predictions[participant] = pd.concat([locations_predictions[participant], pd.DataFrame(
                    {"location_name": location_name, "latitude": latitude, "longitude": longitude})])
            else:
                locations_predictions[participant] = pd.DataFrame(
                    {"location_name": location_name, "latitude": latitude, "longitude": longitude})

        return locations_predictions



#main

#region delete accuracies less than 100m,  missing values in GPS data, and duplicates: for entire GPS dataset and event GPS dataset
dataset_paths = [["GPS-all", "/Users/benediktjordan/Documents/MTS/Iteration01/Data/datasets/locations_all_JSON-transformed_merged-participantIDs.csv"],
                    ["GPS-events_only-active-smartphone-sessions-yes", "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/xmin_around_events_only_active_smartphonesessions/locations_esm_timeperiod_5 min.csv_JSONconverted_only_active_smartphone_sessions.pkl"]]
path_storage = "/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/remove_GPSaccuracy/"
min_gps_accuracy = 100
for path in dataset_paths:
    print("started with " + path[0] + " dataset")
    if path[1].endswith(".pkl"):
        df_locations = pd.read_pickle(path[1])
    else:
        df_locations = pd.read_csv(path[1])

    # delete duplicates in timestamp, device_id, latitude, longitude
    print("number of rows before dropping duplicates: " + str(len(df_locations)))
    # change column name "loc_timestamp" into "timestamp" in case it exists
    if "loc_timestamp" in df_locations.columns:
        df_locations = df_locations.rename(columns={"loc_timestamp": "timestamp"})
    df_locations = df_locations.drop_duplicates(subset=["timestamp", "loc_device_id", "loc_double_latitude", "loc_double_longitude"])
    print("number of rows after dropping duplicates: " + str(len(df_locations)))

    # drop rows which have less accuracy than threshold
    print("number of rows before deleting accuracies: " + str(len(df_locations)))
    df_locations, df_analytics = GPS_computations.GPS_delete_accuracy(df_locations, min_gps_accuracy)
    print("number of rows after deleting accuracies: " + str(len(df_locations)))

    df_locations.to_pickle(path_storage + path[0] + "_min_accuracy_" + str(min_gps_accuracy) + ".pkl")
    df_analytics.to_csv(path_storage + path[0] + "_min_accuracy_" + str(min_gps_accuracy) + "_analytics.csv")

#endregion

#region compute places for each participant
#region compute places
distance_threshold = 0.1 # in km
min_gps_accuracy = 100
df_locations_all = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/remove_GPSaccuracy/GPS-all_min_accuracy_" + str(min_gps_accuracy) + ".pkl")
path_storage = "/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/places/"
#TODO check where "ESM_timestamp" column was introduced into df_locations_all and remove it again

# reset index: important for later merging the clustered data in again
df_locations_all = df_locations_all.reset_index(drop=True)

#delete unnecessary columns
drop_cols = ["ESM_timestamp", "Unnamed: 0.1", "Unnamed: 0", "0", "1", "3", "loc_label", "loc_provider", "loc_double_speed", "loc_double_bearing"]
df_locations_all = df_locations_all.drop(drop_cols, axis = 1)

#merge participant IDs
df_locations_all = Merge_Transform.merge_participantIDs(df_locations_all, users, device_id_col = None, include_cities = False)# merge participant IDs

#testarea: check how many records exist per participant
for participant in df_locations_all["device_id"].unique():
    df_participant = df_locations_all[df_locations_all["device_id"] == participant]
    print("participant " + str(participant) + " has " + str(len(df_participant)) + " records")

#testarea: check how many unique GPS locations exist per participant
for participant in df_locations_all["device_id"].unique():
    df_participant = df_locations_all[df_locations_all["device_id"] == participant]
    # delete duplicates in GPS location
    df_participant = df_participant.drop_duplicates(subset=["latitude", "longitude"], keep="first")
    print("participant " + str(participant) + " has " + str(len(df_participant)) + " GPS locations")

# iterate through participants and compute places
df_analytics_all = pd.DataFrame()
for participant in tqdm(df_locations_all["device_id"].unique()):
    print("start with participant " + str(participant))

    #check if this participant has been already computed
    if os.path.exists(path_storage + "INTERMEDIATE_GPS_min-GPS-accuracy-" + str(min_gps_accuracy) + "_df-locations-including-clusters_participant-" +
                                 str(participant) + ".pkl"):
        print("participant " + str(participant) + " already computed")
        df_locations_all = pd.read_pickle(path_storage + "INTERMEDIATE_GPS_min-GPS-accuracy-" + str(min_gps_accuracy) + "_df-locations-including-clusters_participant-" +
                                 str(participant) + ".pkl")
        df_analytics_all = pd.read_csv(path_storage + "INTERMEDIATE_GPS_min-GPS-accuracy-" + str(min_gps_accuracy) + "_df_analytics_participant-" +
                        str(participant) +".csv")
        continue

    df_participant = df_locations_all[df_locations_all["device_id"] == participant]

    # check if participant has more than 40000 unique GPS locations: if so, split dataframe (otherwise computationally too expensive)
    df_participant_unique = df_participant.drop_duplicates(subset=["latitude", "longitude"], keep="first")
    if len(df_participant_unique) > 40000:
        print("participant " + str(participant) + " has more than 40000 unique GPS locations")
        # iterate through chunks of 40000 GPS locations through df_participant
        for i in range(0, len(df_participant_unique), 40000):
            print("start with chunk " + str(i) + " of participant " + str(participant))
            # get ID of last value in chunk
            df_participant_unique_chunk = df_participant_unique.iloc[i:i+40000]
            # set first value
            if i == 0:
                # take as first value the first value in df_participant
                first_value = df_participant.index[0]
            else:
                first_value = last_value-1

            # set last value: if it is the last iteration, take the last value in df_participant
            if i >= (len(df_participant_unique)-40000):
                last_value = df_participant.index[-1]
            else:
                last_value = df_participant_unique_chunk.index[-1]

            #create df_participant_chunk from df_participant
            df_participant_chunk = df_participant.loc[first_value:last_value]
            df_label, df_analytics, = GPS_clustering.agg_clustering_computing_centroids(df_participant_chunk, distance_threshold)
            # add chunk identifier column to df_label and df_analytics
            df_label["chunk"] = i
            df_analytics["chunk"] = i

            # merge df_label into df_locations_all
            df_locations_all.loc[df_label.index, "cluster_label"] = df_label["cluster_label"]
            df_locations_all.loc[df_label.index, "cluster_latitude_mean"] = df_label["cluster_latitude_mean"]
            df_locations_all.loc[df_label.index, "cluster_longitude_mean"] = df_label["cluster_longitude_mean"]
            df_locations_all.loc[df_label.index, "chunk"] = df_label["chunk"]

            # concatenate df_analytics
            df_analytics_all = pd.concat([df_analytics_all, df_analytics])

            #double check
            print("Number of NaN values in cluster_label: " + str(df_label["cluster_label"].isna().sum()))


    else:
        df_label, df_analytics, = GPS_clustering.agg_clustering_computing_centroids(df_participant, distance_threshold)

        # merge df_label into df_locations_all
        df_locations_all.loc[df_label.index, "cluster_label"] = df_label["cluster_label"]
        df_locations_all.loc[df_label.index, "cluster_latitude_mean"] = df_label["cluster_latitude_mean"]
        df_locations_all.loc[df_label.index, "cluster_longitude_mean"] = df_label["cluster_longitude_mean"]

        #concatenate df_analytics
        df_analytics_all = pd.concat([df_analytics_all, df_analytics])

    #save intermediate files
    df_analytics_all.to_csv(path_storage + "INTERMEDIATE_GPS_min-GPS-accuracy-" + str(min_gps_accuracy) + "_df_analytics_participant-" +
                        str(participant) +".csv")
    # save df_locations_all as pickle
    df_locations_all.to_pickle(path_storage + "INTERMEDIATE_GPS_min-GPS-accuracy-" + str(min_gps_accuracy) + "_df-locations-including-clusters_participant-" +
                                 str(participant) + ".pkl")
df_locations_all.to_pickle(path_storage + "GPS_min-GPS-accuracy-" + str(min_gps_accuracy) + "_df-locations-including-clusters.pkl")
df_analytics_all.to_csv(path_storage + "GPS_min-GPS-accuracy-" + str(min_gps_accuracy) + "_df_analytics.csv")

# analytics section: how many clusters are there for each participant?
df_analytics = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/places/GPS_min-GPS-accuracy-100_df_analytics.csv")
## delete records with anything else than NaN or 0.0 in chunk column
df_analytics = df_analytics[df_analytics["chunk"].isna() | (df_analytics["chunk"] == 0.0)]
for participant in df_analytics["participant"].unique():
    print("Number of clusters for participant " + str(participant) + ": " + str(len(df_analytics[df_analytics["participant"] == participant]["cluster_label"].unique())))
## get the average number of clusters per participant
print("Average number of clusters per participant: " + str(df_analytics["cluster_label"].nunique()/len(df_analytics["participant"].unique())))

## get for every participant how many clusters contain 90% of all "number_points" values
for participant in df_analytics["participant"].unique():
    df_participant = df_analytics[df_analytics["participant"] == participant]
    # sort df_participant by number_points
    df_participant = df_participant.sort_values(by="number_points", ascending=False)
    # get the 90% of the sum of all number_points
    sum_90_percent = df_participant["number_points"].sum()*0.9
    # get the number of clusters which contain 90% of all number_points
    sum_90_percent_clusters = 0
    for i in range(len(df_participant)):
        sum_90_percent_clusters += df_participant.iloc[i]["number_points"]
        if sum_90_percent_clusters >= sum_90_percent:
            print("Participant " + str(participant) + " has " + str(i+1) + " clusters which contain 90% of all number_points.")
            break
mean = np.mean([29, 25, 79, 10, 33, 29])
print(mean)

## get for every participant, how many clusters contain more than 600 "number_points" values
get_mean = []
for participant in df_analytics["participant"].unique():
    df_participant = df_analytics[df_analytics["participant"] == participant]
    print("Participant " + str(participant) + " has " + str(len(df_participant[df_participant["number_points"] > 10800])) + " clusters for which the participant staid at leas three hours.")
    get_mean.append(len(df_participant[df_participant["number_points"] > 10800]))
print("Mean, median, min, max and standard deviation of clusters which contain more than three hours: " + str(np.mean(get_mean)) + ", " + str(np.median(get_mean)) + ", " + str(np.min(get_mean)) + ", " + str(np.max(get_mean)) + ", " + str(np.std(get_mean)))
#endregion

#region only keep 30 most frequent clusters and add column "number of clusters" and "number of total records"
# only keep clusters which are among 30 most frequent clusters (for every participant and every chunk)
df_locations = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/places/GPS_min-GPS-accuracy-100_df-locations-including-clusters.pkl")
df_analytics = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/places/GPS_min-GPS-accuracy-100_df_analytics.csv")
# create column "number of records for participant"
df_locations["records_total_for_participant"] = np.nan
for participant in df_locations["device_id"].unique():
    print("start with participant " + str(participant) + "...")
    df_participant = df_locations[df_locations["device_id"] == participant]
    # get number of records for participant
    number_of_records_for_participant = len(df_participant)
    df_locations.loc[df_participant.index, "records_total_for_participant"] = number_of_records_for_participant
    # check if participant has any values apart from NaN in chunk
    ## these are participants for which the clusters have been computed in chunks
    ## for them: the most frequent 30 clusters PER CHUNK are computed
    if df_participant["chunk"].isna().sum() != len(df_participant):
        # get list of chunks
        chunks = df_participant["chunk"].unique()
        # iterate through chunks and delete records not in 30 most frequent clusters
        for chunk in chunks:
            print("start with chunk " + str(chunk) + " of participant " + str(participant))
            # get df_participant_chunk
            df_participant_chunk = df_participant[df_participant["chunk"] == chunk]
            # get 30 most frequent clusters
            df_participant_chunk_most_frequent_clusters = df_participant_chunk["cluster_label"].value_counts().head(30)
            # get list of records not in most frequent clusters
            df_participant_chunk_not_in_most_frequent_clusters = df_participant_chunk[~df_participant_chunk["cluster_label"].isin(df_participant_chunk_most_frequent_clusters.index)]
            # delete records not in most frequent clusters in df_locations
            df_locations.drop(df_participant_chunk_not_in_most_frequent_clusters.index, inplace=True)
    else:
        # get 30 most frequent clusters
        df_participant_most_frequent_clusters = df_participant["cluster_label"].value_counts().head(30)
        # get list of records not in most frequent clusters
        df_participant_not_in_most_frequent_clusters = df_participant[~df_participant["cluster_label"].isin(df_participant_most_frequent_clusters.index)]
        # delete records not in most frequent clusters in df_locations
        df_locations.drop(df_participant_not_in_most_frequent_clusters.index, inplace=True)
df_locations.to_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/places/GPS_min-GPS-accuracy-100_df-locations-including-clusters_OnlyBiggest30Clusters.pkl")
#endregion

#testarea: why are there NaN vlaues in cluster_label?
## how many values for participant
len(df_locations_all[df_locations_all["device_id"] == participant])
df_test2 = df_locations_all[df_locations_all["device_id"] == participant]
# check how many NaN values in cluster_labels
df_test2["cluster_label"].isna().sum()
## check in fixErrors if there are any NaN values after computing clusters for participant 3


#region merge clusters for participants which have been computed in chunks
# solve problem for places which have been computed in chunks
## Description: for every participant for whom the places were computed in chunks:
### 1. for every chunk: for every cluster in this chunk: the distances between the cluster_means and all cluster means of clusers in other chunks are computed.
### 2. if the distance is smaller than 100 meter: the distance between all points in this cluster and the close cluster are computed
### 3, if the distance between the "farthes away" points is less than 150 meters: the cluster is merged into the close cluster
df_locations = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/places/GPS_min-GPS-accuracy-100_df-locations-including-clusters_OnlyBiggest30Clusters.pkl")
participant_list = df_locations.groupby("device_id").filter(lambda x: len(x["chunk"].unique()) > 1)["device_id"].unique()
df_locations_merged, df_analytics  = GPS_clustering.merge_clusters_of_chunks(df_locations, participant_list)
df_locations_merged.to_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/places/GPS_min-GPS-accuracy-100_df-locations-including-clusters_OnlyBiggest30Clusters_ClustersFromChunksMerged.pkl")
df_analytics.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/places/GPS_min-GPS-accuracy-100_df-locations-including-clusters_OnlyBiggest30Clusters_ClustersFromChunksMerged_analytics.csv")
#endregion
#endregion

#region compute features for places

#region calculate features for GPS data: this is necessary to compute features for places
df_locations = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/places/GPS_min-GPS-accuracy-100_df-locations-including-clusters_OnlyBiggest30Clusters_ClustersFromChunksMerged.pkl")
df_results = pd.DataFrame()

## delete NaNs in cluster_label
#TODO find out why there are NaNs in cluster_label: this problem is solved, but new clustering not executed yet; its minor, therefore can also proceed with this
print("number of NaNs in cluster_label: " + str(df_locations["cluster_label"].isna().sum()))
df_NaN = df_locations[df_locations["cluster_label"].isna()]
df_locations = df_locations.dropna(subset=["cluster_label"])

## add local timezone column
## create from users a dictionary with device_id as key and timezone as value
time_zones = {}
for index, row in users.iterrows():
    time_zones[row["new_ID"]] = pytz.timezone(row["Timezone"])
## add "timestamp_local" column
df_locations['timestamp_local'] = df_locations.apply(lambda x: convert_timestamp_to_local_timezone(x['loc_timestamp'], time_zones[x['device_id']]), axis=1)
# change format from object to datetime for timestamp_local
df_locations["timestamp_local"] = df_locations["timestamp_local"].astype(str)
df_locations["timestamp_local"] = df_locations["timestamp_local"].str.rsplit("+", expand=True)[0]
df_locations["timestamp_local"] = pd.to_datetime(df_locations["timestamp_local"])

#TODO the assumption of one timezone for every participant is not entirely correct: for one participant there ws sometimes a different timezone used (Amsterdam instead of London)
#TODO: you can check with this code:
## get timezones for participants
#df_timezones = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/xmin_around_events/timezone_esm_timeperiod_5 min.csv_JSONconverted.pkl")
#df_timezones = Merge_Transform.merge_participantIDs(df_timezones, users, device_id_col = None, include_cities = False)# merge participant IDs
#df_timezones = df_timezones[["device_id", "tim_timezone"]].drop_duplicates(subset=["device_id", "tim_timezone"], keep="first")

## calculate features
df_results = pd.DataFrame()
for participant in df_locations["device_id"].unique():
    df_participant = df_locations[df_locations["device_id"] == participant]
    print("start with participant " + str(participant) + " (" + str(len(df_participant)) + " rows)")

    # delete rows which have same timestamp, device_id, latitude, and longitude
    df_participant = df_participant.drop_duplicates(subset=["loc_timestamp", "device_id", "latitude", "longitude"], keep="first")

    # convert unix time to datetime (loc_timestamp)
    df_participant["loc_timestamp"] = pd.to_datetime(df_participant["loc_timestamp"], unit="ms")
    df_participant = df_participant.rename(columns={"loc_timestamp": "timestamp"})

    ## calculate number of days for which GPS data is available
    df_participant["days_with_data_count"] = FeatureExtraction_GPS.gps_features_count_days_with_enough_data(df_participant, min_gps_points = 60)

    ## create weekday column
    df_participant["weekday"] = df_participant["timestamp_local"].dt.weekday
    ## create weekend vs. workday column
    df_participant["weekend_weekday"] = df_participant["weekday"].apply(lambda x: "weekend" if x in [5,6] else "weekday")
    ## create hour_of_day column
    df_participant["hour_of_day"] = df_participant["timestamp_local"].dt.hour
    ## create time_of_day column
    df_participant["time_of_day"] = df_participant["hour_of_day"].apply(lambda x: "morning" if x in [6,7,8,9,10,11] else "afternoon" if x in [12,13,14,15,16,17] else "evening" if x in [18,19,20,21,22,23] else "night" if x in [0,1,2,3,4,5] else "unknown")

    # create stays and compute stays features
    df = df_participant.copy()
    freq = 1 # in Hz; frequency of the sensor data
    duration_threshold= 60 #minimum duration of a stay in seconds
    min_records_fraction=0.5 #minimum fraction of records which have to exist in a chunk to be considered as a stay
    df_participant = FeatureExtraction_GPS.compute_stays_and_stays_features(df_participant, freq, duration_threshold, min_records_fraction)

   # convert cluster_label to int
    df_participant["cluster_label"] = df_participant["cluster_label"].astype(int)

    print("finished with participant " + str(participant) + " (" + str(len(df_participant)) + " rows)")
    # concatenate into df_results
    df_results = pd.concat([df_results, df_participant])
#TODO find out why I didn´t drop duplicates in any of the steps above but only here; and why are there so many duplicates?
df_results.to_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/places/GPS_features/GPS_min-GPS-accuracy-100_df-locations-including-clusters_OnlyBiggest30Clusters_ClustersFromChunksMerged_withGPSFeatures.pkl")



#endregion

#region calculate features for places
df_locations = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/places/GPS_features/GPS_min-GPS-accuracy-100_df-locations-including-clusters_OnlyBiggest30Clusters_ClustersFromChunksMerged_withGPSFeatures.pkl")
frequency = 1 # in Hz; frequency of the sensor data
timeslot_lists = [["hour_of_day", [0,1,2,3,4,5,6, 7,8,9,10,11,12, 13,14,15,16,17,18, 19,20,21,22,23]],
                    ["time_of_day", ["morning", "afternoon", "evening", "night"]],
                    ["weekday", [0,1,2,3,4,5,6]],
                    ["weekend_weekday", ["weekend", "weekday"]]]

### create columns for timeslots
columns = []
for timeslot_list in timeslot_lists:
    timeslot_type = timeslot_list[0]
    timslots = timeslot_list[1]
    for timeslot in timslots:
        columns.append(timeslot_type + "_" + str(timeslot) + "_arrive_percentage")
        columns.append(timeslot_type + "_" + str(timeslot) + "_arrive_percentage_trusted")
        columns.append(timeslot_type + "_" + str(timeslot) + "_leave_percentage")
        columns.append(timeslot_type + "_" + str(timeslot) + "_leave_percentage_trusted")
        columns.append(timeslot_type + "_" + str(timeslot) + "_intersecting_percentage")
        columns.append(timeslot_type + "_" + str(timeslot) + "_intersecting_percentage_trusted")
        columns.append(timeslot_type + "_" + str(timeslot) + "_time_fraction")
        columns.append(timeslot_type + "_" + str(timeslot) + "_time_fraction_trusted")
        columns.append(timeslot_type + "_" + str(timeslot) + "_time_per_day")
        columns.append(timeslot_type + "_" + str(timeslot) + "_time_per_day_trusted")
further_columns = ["device_id", "place", "total_records_in_cluster", "total_records_fraction", "total_records_of_biggest_clusters_fraction", "visits_per_day", "visits_per_day_trusted",
                                    "visits_fraction", "visits_fraction_trusted", "time_per_day", "time_per_day_trusted",
                                    "stay_duration_mean", "stay_duration_mean_trusted", "stay_duration_max", "stay_duration_max_trusted",
                                    "fraction_of_time_spent_at_place", "fraction_of_time_spent_at_place_trusted"]
columns_list = further_columns + columns

### only keep 15 biggest clusters
for participant in df_locations["device_id"].unique():
    df_participant = df_locations[df_locations["device_id"] == participant]
    print("Participant " + str(participant) + " has " + str(len(df_participant["cluster_label"].unique())) + " clusters")
    # get the index of the all clusters except the 15 biggest
    index_to_drop = df_participant["cluster_label"].value_counts().index[15:]
    # drop all rows with these clusters for this participant
    df_locations = df_locations.drop(df_locations[(df_locations["device_id"] == participant) & (df_locations["cluster_label"].isin(index_to_drop))].index)
    print("Participant " + str(participant) + " has " + str(len(df_locations[df_locations["device_id"] == participant]["cluster_label"].unique())) + " clusters left")

### compute features
df_results = pd.DataFrame(columns= columns_list)
for participant in tqdm(df_locations["device_id"].unique()):

    # compute features
    print("computing features for participant " + str(participant))
    df_participant = df_locations[df_locations["device_id"] == participant]
    cluster_counter = 1
    for place in df_participant["cluster_label"].unique():
        print("computing features for place (" + str(cluster_counter) + "/" + str(len(df_participant["cluster_label"].unique())) + ")")
        cluster_counter += 1

        line_number = len(df_results)
        df_results.loc[line_number] = np.nan

        # add device_id and place
        df_results.loc[line_number]["device_id"] = participant
        df_results.loc[line_number]["place"] = place

        # add total records in the cluster
        df_results.loc[line_number]["total_records_in_cluster"] = len(df_participant[df_participant["cluster_label"] == place])

        # add fraction of total records which are at this place: number of total records for each participant is in column records_total_for_participant
        df_results.loc[line_number]["total_records_fraction"] = len(df_participant[df_participant["cluster_label"] == place]) / df_participant["records_total_for_participant"].iloc[0]

        # add fraction of total records of 30 biggest clusters which are at this place
        df_results.loc[line_number]["total_records_of_biggest_clusters_fraction"] = len(df_participant[df_participant["cluster_label"] == place]) / len(df_participant)

        # compute visits_per_day for place
        df_results.loc[line_number]["visits_per_day"], df_results.loc[line_number]["visits_per_day_trusted"] = FeatureExtraction_GPS.gps_features_places_visits_per_day(df_participant, participant, place)

        # compute visits_fraction for place
        df_results.loc[line_number]["visits_fraction"], df_results.loc[line_number]["visits_fraction_trusted"] = FeatureExtraction_GPS.gps_features_places_visits_fraction(df_participant, participant, place)

        # compute time_per_day for place
        df_results.loc[line_number]["time_per_day"], df_results.loc[line_number]["time_per_day_trusted"] = FeatureExtraction_GPS.gps_features_places_time_per_day(df_participant, participant, place)

        # stay_duration_max and mean: normal and trusted
        df_results.loc[line_number]["stay_duration_max"] = df_participant[(df_participant["cluster_label"] == place) & (df_participant["device_id"] == participant)]["stay_duration"].max()
        df_results.loc[line_number]["stay_duration_max_trusted"]  = df_participant[(df_participant["cluster_label"] == place) & (df_participant["device_id"] == participant)]["stay_duration_trusted"].max()
        df_results.loc[line_number]["stay_duration_mean"]  = df_participant[(df_participant["cluster_label"] == place) & (df_participant["device_id"] == participant)]["stay_duration"].mean()
        df_results.loc[line_number]["stay_duration_mean_trusted"]  = df_participant[(df_participant["cluster_label"] == place) & (df_participant["device_id"] == participant)]["stay_duration_trusted"].mean()

        # fraction of time spent at this place (of all time spent at all places)
        df_results.loc[line_number]["fraction_of_time_spent_at_place"] = df_participant[(df_participant["cluster_label"] == place) & (df_participant["device_id"] == participant)]["stay_duration"].sum() / df_participant[(df_participant["device_id"] == participant)]["stay_duration"].sum()

        # trusted fraction of time spent at this place (of all time spent at all places)
        df_results.loc[line_number]["fraction_of_time_spent_at_place_trusted"] = \
        df_participant[(df_participant["cluster_label"] == place) & (df_participant["device_id"] == participant)][
            "stay_duration"].sum() / df_participant[(df_participant["device_id"] == participant)]["stay_duration_trusted"].sum()

        # calculate features for different timeslots
        timeslot_counter = 1
        # create number of timeslots as all elements withing the second level of the list
        number_of_timeslots = sum([len(timeslot_list[1]) for timeslot_list in timeslot_lists])
        for timeslot_list in timeslot_lists:
            timeslot_type = timeslot_list[0]
            for timeslot in timeslot_list[1]:
                print("computing features for timeslot " + str(timeslot_counter) + "/" + str(number_of_timeslots))
                timeslot_counter += 1

                # compute arrive, leave, and intersecting percentage for place
                df_results.loc[line_number][timeslot_type + "_" + str(timeslot) + "_arrive_percentage"], df_results.loc[line_number][
                    timeslot_type + "_" + str(timeslot)  + "_arrive_percentage_trusted"], \
                    df_results.loc[line_number][timeslot_type + "_" + str(timeslot)  + "_leave_percentage"], df_results.loc[line_number][
                    timeslot_type + "_" + str(timeslot)  + "_leave_percentage_trusted"], \
                    df_results.loc[line_number][timeslot_type + "_" + str(timeslot)  + "_intersecting_percentage"], df_results.loc[line_number][
                    timeslot_type + "_" + str(timeslot)  + "_intersecting_percentage_trusted"] = FeatureExtraction_GPS.compute_arrive_leave_intersecting_percentage(df_participant,
                                                                                                              participant,
                                                                                                              place,
                                                                                                              timeslot,
                                                                                                              timeslot_type)
                # compute time fraction and time per day
                df_results.loc[line_number][timeslot_type + "_" + str(timeslot) + "_time_fraction"], \
                    df_results.loc[line_number][timeslot_type + "_" + str(timeslot) + "_time_fraction_trusted"], \
                    df_results.loc[line_number][timeslot_type + "_" + str(timeslot) + "_time_per_day"], df_results.loc[line_number][
                    timeslot_type + "_" + str(timeslot) + "_time_per_day_trusted"] = FeatureExtraction_GPS.compute_time_fraction_time_per_day(df_participant, participant, place, timeslot, timeslot_type, frequency)
    #save intermediate file
    df_results.to_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/places/places_features/INTERMEDIATE_GPS_min-GPS-accuracy-100_df-locations-including-clusters_OnlyBiggest30Clusters_ClustersFromChunksMerged_withPlacesFeatures_participant-" + str(participant) + ".pkl")
df_results.to_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/places/places_features/GPS_min-GPS-accuracy-100_df-locations-including-clusters_OnlyBiggest30Clusters_ClustersFromChunksMerged_PlacesFeatures.pkl")

# delete all places which don´t have at least 1 stay
print("number of places before deleting places with no stay: " + str(len(df_results)))
df_results = df_results[df_results["stay_duration_mean"] > 0]
print("number of places after deleting places with no stay: " + str(len(df_results)))

# calculate features which put other place-features in a relationship: rank ascent and rank descent
## delete "device_id" and "place" from column_list
columns_list.remove("device_id")
columns_list.remove("place")
df_results = FeatureExtraction_GPS.gps_features_places_rank_ascent_descent(df_results, columns_list)
#TODO double check the intermediate results with analytics dataframe or print functions

df_results.to_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/places/places_features/places_min-GPS-accuracy-100_df-locations-including-clusters_OnlyBiggest30Clusters_ClustersFromChunksMerged_withPlacesFeatures_DeletedPlacesWithoutStays.pkl")
#endregion

#testarea: double check places feature calculation
df_test = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/places/places_features/INTERMEDIATE_GPS_min-GPS-accuracy-100_df-locations-including-clusters_OnlyBiggest30Clusters_ClustersFromChunksMerged_withPlacesFeatures_participant-2.pkl")







#endregion

#region get GPS locations for home and office for each participant
min_gps_accuracy = 100
only_active_smartphone_sessions = "yes"
distance_threshold = 0.1 #in km; maximum distance between most distant points in any cluster
df_locations_events = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/remove_GPSaccuracy/GPS-events_only-active-smartphone-sessions-"+ only_active_smartphone_sessions +"_min_accuracy_" + str(min_gps_accuracy) +".pkl")
dict_label = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled_dict.pkl")
label_column_name = "label_location"
classes = ["at home", "in the office"]
path_storage = "/Users/benediktjordan/Documents/MTS/Iteration01/location/data_preparation/map_classes_to_locations_and_wifi/"

# for every event: only keep first GPS record afterwards
df_locations_events = GPS_computations.first_sensor_record_after_event(df_locations_events)

#label sensor data & drop NaNs in label column
df_locations_events = labeling_sensor_df(df_locations_events, dict_label, label_column_name, ESM_identifier_column = "ESM_timestamp")
df_locations_events = Merge_Transform.merge_participantIDs(df_locations_events, users, device_id_col = None, include_cities = False)# merge participant IDs
df_locations_events = df_locations_events.dropna(subset=[label_column_name])

# for each participant and class: cluster GPS points and take dominant cluster as location
df_analytics_all = pd.DataFrame(columns=["participant", "class", "cluster_label", "number_points", "latitude_mean", "longitude_mean"])
for participant in df_locations_events["device_id"].unique():
    print("Started with participant " + str(participant) + "")
    df_participant = df_locations_events[df_locations_events["device_id"] == participant]
    for label in classes:
        df_label = df_participant[df_participant[label_column_name] == label]
        # if df_label is empty, continue with next label
        if df_label.empty:
            print("no label " + label + " for participant " + str(participant) + "")
            continue

        # if only one value in df_label, no clustering needed
        if len(df_label) == 1:
            df_locations_events.loc[df_label.index, "cluster_label"] = 0
            df_locations_events.loc[df_label.index, "cluster_latitude_mean"] = df_label["latitude"].iloc[0]
            df_locations_events.loc[df_label.index, "cluster_longitude_mean"] = df_label["longitude"].iloc[0]
            # add label to df_analytics
            df_analytics = pd.DataFrame({"participant": [participant], "class": [label], "cluster_label": [0], "number_points": [1],
                                            "latitude_mean": [df_label["latitude"].iloc[0]], "longitude_mean": [df_label["longitude"].iloc[0]]})

        # cluster GPS locations
        else:
            df_label, df_analytics, dominant_cluster_latitude, dominant_cluster_longitude = agg_clustering_computing_centroids(df_label, distance_threshold)

            # replace records in df_locations_events with the records in df_label
            df_locations_events.loc[df_label.index, "cluster_label"] = df_label["cluster_label"]
            df_locations_events.loc[df_label.index, "cluster_latitude_mean"] = df_label["cluster_latitude_mean"]
            df_locations_events.loc[df_label.index, "cluster_longitude_mean"] = df_label["cluster_longitude_mean"]

            # add label to df_analytics
            df_analytics["class"] = label

        #concatenate df_analytics to df_analytics_all
        df_analytics_all = pd.concat([df_analytics_all, df_analytics])



        print(agg_clustering.labels_)
df_analytics_all["participant"] = df_analytics_all["participant"].astype(int)
df_analytics_all.to_csv(path_storage + "GPS_only_active_smartphone_sessions" + only_active_smartphone_sessions + "_min-GPS-accuracy-" + str(min_gps_accuracy) +
          "_df_analytics.csv")
df_locations_events.to_csv(path_storage + "GPS_only_active_smartphone_sessions" + only_active_smartphone_sessions + "_min-GPS-accuracy-" + str(min_gps_accuracy) +
            "_df-locations-including-clusters.csv")

# create dictionary which maps class to dominant cluster location
dict_class_locations = {}
for participant in df_analytics_all["participant"].unique():
    dict_class_locations[participant] = {}
    for label in df_analytics_all["class"].unique():
        df_participant_label = df_analytics_all[(df_analytics_all["participant"] == participant) & (df_analytics_all["class"] == label)]
        # get the dominant cluster (cluster with most points)
        df_participant_label = df_participant_label[df_participant_label["number_points"] == df_participant_label["number_points"].max()]
        if df_participant_label.empty:
            continue
        dict_class_locations[participant][label] = {"latitude": df_participant_label["latitude_mean"].iloc[0],
                                                    "longitude": df_participant_label["longitude_mean"].iloc[0]}
with open(path_storage + "GPS_only_active_smartphone_sessions" + only_active_smartphone_sessions + "_min-GPS-accuracy-" + str(min_gps_accuracy) +
          "_dict_mapping_class_locations.pkl", "wb") as f:
    pickle.dump(dict_class_locations, f)


#endregion








#region testsection: implement agglomerative clustering including centroid linkage and distance_threshold
# using haversine distance

import numpy as np
import haversine
from sklearn.cluster import AgglomerativeClustering

latitude = [51.33789164632932, 51.3371901217961, 51.33661165641516, 51.33628439344723, 51.334593788165506,51.33428040578081,51.33343309949297,51.33295733241303 ]
longitude = [12.360512, 12.360511998172617,12.362259201795089, 12.362676366389088,12.354203442831327,12.354549792864173,12.355702895198947,12.356312081338071]
#create np array
X = np.array(list(zip(latitude, longitude)))

df_label, df_analytics, = GPS_clustering.agg_clustering_computing_centroids(df_participant, distance_threshold)


distances = [0, 100, 190, 230]
51.3371901217961, 12.361434653994957
51.33789164632932, 12.360511998172617
51.33661165641516, 12.362259201795089
51.33628439344723, 12.362676366389088

51.334593788165506, 12.354203442831327
51.33428040578081, 12.354549792864173
51.33343309949297, 12.355702895198947
51.33295733241303, 12.356312081338071

## standard algorithm
#calculate distance matrix using geodeisc distance


agg_clustering = AgglomerativeClustering(linkage="complete",n_clusters=None, distance_threshold=0.3, metric = "precomputed").fit(dist_matrix)
print(agg_clustering.labels_)




# function to compute haversine distance
def haversine_distance(point1, point2, R=6371):
    lat1, lon1 = radians(point1[0]), radians(point1[1])
    lat2, lon2 = radians(point2[0]), radians(point2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

class CentroidLinkageAgglomerativeClustering:
    def __init__(self, n_clusters, linkage='centroid', distance_threshold=None, haversine=False):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.distance_threshold = distance_threshold
        self.haversine = haversine

    def fit(self, X):
        n_samples, n_features = X.shape

        # Initialize cluster labels
        labels = np.arange(n_samples)

        # Initialize cluster centroids
        centroids = np.array([X[i] for i in range(n_samples)])

        # Initialize cluster sizes
        sizes = np.ones(n_samples, dtype=np.int64)

        # Repeat until the desired number of clusters is reached
        while len(np.unique(labels)) > self.n_clusters:
            # Compute distances between all pairs of centroids
            distances = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    if self.haversine:
                        distances[i, j] = haversine.distance(centroids[i], centroids[j])
                    else:
                        distances[i, j] = np.linalg.norm(centroids[i] - centroids[j])

            # Find the pair of clusters with the smallest distance
            min_index = np.argmin(distances)
            i, j = np.unravel_index(min_index, distances.shape)

            # Check if the distance is below the threshold (if specified)
            if self.distance_threshold and distances[i, j] > self.distance_threshold:
                break

            # Merge the two clusters
            labels[labels == i] = j
            centroids[j] = (sizes[i] * centroids[i] + sizes[j] * centroids[j]) / (sizes[i] + sizes[j])
            sizes[j] += sizes[i]

            # Remove the merged cluster
            centroids = np.delete(centroids, i, axis=0)
            sizes = np.delete(sizes, i)

        self.labels_ = labels
        self.cluster_centers_ = centroids

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
#endregion













#endregion

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


#region calculate distance between labels and most frequent locations
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
#endregion





#region OUTDATED: modal approach: find the most frequent location during every night of the participants: take for every night the location with the highest frequency
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
#endregion

