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

# class contains methods to implement the agglomerative clustering algorithm
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


#region OUTDATES: first trials
#test area
# print value counts in chunks for every device_id
print("start printing value counts")
for device_id in df_locations["device_id"].unique():
    print("participant " + str(device_id))
    print(df_locations[df_locations["device_id"] == device_id]["chunk"].value_counts())
    # print how many unique pairs of latitude and longitude there are
    print("number of unique pairs of latitude and longitude: " + str(len(df_locations[df_locations["device_id"] == device_id][["latitude", "longitude"]].drop_duplicates())))

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

#region testsection: implement agglomerative clustering including centroid linkage and distance_threshold
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

#endregion