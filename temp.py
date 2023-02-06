#region import
import pickle
#import tensorflow_decision_forests as tfdf
import datetime
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import math
import time
import matplotlib.pyplot as plt
import json

# computing distance
import geopy.distance

#Classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

#evaluation
from sklearn.metrics import balanced_accuracy_score

#nested CV
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
# import classification report
from sklearn.metrics import classification_report

#Feature Importance
import shap

#Statistical Testing
from scipy.stats import binom_test
from sklearn.model_selection import permutation_test_score

#Keras Tuner
import keras
import keras_tuner as kt
from kerastuner.tuners import RandomSearch
from kerastuner.tuners import BayesianOptimization
from kerastuner.tuners import Hyperband

# visualization
import seaborn as sns

# for keeping track
from tqdm import tqdm



try:
  from wurlitzer import sys_pipes
except:
  from colabtools.googlelog import CaptureLog as sys_pipes
#endregion

#temporary
#df_temp = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_preparation/features/temp/df_final_14.csv")
#save df_temp as pkl with highest protocoll
#with open("/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_preparation/features/temp/df_final_14_2.pkl", 'wb') as f:
#    pickle.dump(df_temp, f, pickle.HIGHEST_PROTOCOL)

class Merge_and_Impute:

    #merge timeseries or features
    # NOTE_
    ## timedelta HAS TO BE a string in format "10s", otherwise it doesnt work
    ## merging is done using the setting "nearest" which means the nearest timestamp is taken (doesnt matter if before or after)
    def merge(df_base, df_tomerge, sensor_merge, timedelta, columns_to_delete, path_intermediate_files = None, add_prefix_to_merged_columns = False):

        df_final = pd.DataFrame()

        # harmonize column names
        ## if there is a column named "2" in df_tomerge, rename it to "device_id"
        if "2" in df_base.columns:
            df_base = df_base.rename(columns={"2": "device_id"})
        if "2" in df_tomerge.columns:
            df_tomerge = df_tomerge.rename(columns={"2": "device_id"})
        if "device_id" not in df_base.columns:
            for col in df_base.columns:
                if "device_id" in col:
                    df_base = df_base.rename(columns={col: "device_id"})
        if "device_id" not in df_tomerge.columns:
            for col in df_tomerge.columns:
                if "device_id" in col:
                    df_tomerge = df_tomerge.rename(columns={col: "device_id"})

        ## if there is a column named "sensor_timestamp", rename it to "timestamp"
        if "sensor_timestamp" in df_base.columns:
            df_base.rename(columns={"sensor_timestamp": "timestamp"}, inplace=True)
        if "sensor_timestamp" in df_tomerge.columns:
            df_tomerge = df_tomerge.rename(columns={"sensor_timestamp": "timestamp"})

        user_count = 1
        total_users = len(df_base['device_id'].unique())
        # iterate through participants and ESM_timestamps
        for user in df_base['device_id'].unique():
            if path_intermediate_files != None:

                # check if for this user, there is already a df_final
                if os.path.exists(path_intermediate_files + "df_final_" + str(user_count) + ".pkl"):
                    #check if also for next user_count, there is a df_final
                    if os.path.exists(path_intermediate_files + "df_final_" + str(user_count + 1) + ".pkl"):
                        print("df_final_" + str(user_count) + ".csv already exists")
                        # if yes: increase user_count and continue
                        user_count += 1
                        continue
                    user_count += 1
                    continue
                # if not: load df_final
                if user_count != 1:
                    print("df_final_" + str(user_count-1) + " will be used now")
                    df_final = pd.read_pickle(path_intermediate_files + "df_final_" + str(user_count - 1) + ".pkl")

            print("User with ID " + str(user) + " is being processed.")
            print("This is user " + str(user_count) + " of " + str(total_users) + ".")
            time_start = time.time()

            df_base_user = df_base[df_base['device_id'] == user]
            df_tomerge_user = df_tomerge[df_tomerge['device_id'] == user]

            # make sure that timestamp columns are in datetime format with using .loc
            df_base_user['timestamp'] = pd.to_datetime(df_base_user['timestamp'])
            df_tomerge_user['timestamp'] = pd.to_datetime(df_tomerge_user['timestamp'])

            # sort dataframes by timestamp
            df_base_user = df_base_user.sort_values(by='timestamp')
            df_tomerge_user = df_tomerge_user.sort_values(by='timestamp')

            # duplicate timestamp column for test purposes and ESM_timestamp column in order to get ESM_timestamp for rows later where only data from df_tomerge is in
            df_tomerge_user['timestamp_merged'] = df_tomerge_user['timestamp'].copy()
            df_tomerge_user['ESM_timestamp_merged'] = df_tomerge_user['ESM_timestamp'].copy()

            # delete columns from df_tomerger_user that dont have to be merged
            df_tomerge_user = df_tomerge_user.drop(columns= columns_to_delete)
            #df_tomerge_user = df_tomerge_user.drop(columns=['device_id'])

            # delete columns "Unnamed: 0", "0", "1" and "2" from df_sensor_user_event: all the information of these
            # columns is already contained in the JSON data
            #df_sensor_user_event = df_sensor_user_event.drop(columns=['Unnamed: 0', '0', '1', '2'])

            #if sensor timeseries are merged: add prefix of three first letters of sensor to column names
            # in order that the "double_values_X" columns of different sensors can be differentiated
            if add_prefix_to_merged_columns == True:
                # add prefix to column names except for "timestamp" and "device_id"
                for col in df_tomerge_user.columns:
                    if col != "timestamp" and col != "device_id" and col != "timestamp_merged" and col != "ESM_timestamp" and col != "ESM_timestamp_merged":
                        df_tomerge_user = df_tomerge_user.rename(columns={col: sensor_merge[:3] + "_" + col})

            # merge dataframes
            df_merged = pd.merge_asof(df_base_user, df_tomerge_user, on= "timestamp",
                                      tolerance=pd.Timedelta(timedelta), direction='nearest')
            # Explanation: this function looks for every entry in the left timeseries if there is a
            # entry in the right timeseries which is max. "timedelta" time apart; direction = "nearest" means
            # that the closest entry is chosen
            # TODO: include functionality so that also sensors with lesser frequency can be merged (i.e.
            #  locations, open_wheather etc.)

            # delete from df_tomerge_user all rows which have a timestamp that is in df_merged["timestamp_merged"]
            df_tomerge_user = df_tomerge_user[~df_tomerge_user['timestamp_merged'].isin(df_merged["timestamp_merged"])]
            print("df_tomerge_user.shape after deleting rows that are already in df_merged: ", df_tomerge_user.shape)

            #concatenate df_merged and df_tomerge_user
            ## Note: this adds to df_merged all the records from df_tomerge_user which couldn´t be merged
            df_merged = pd.concat([df_merged, df_tomerge_user], axis=0)
            print("df_merged.shape after concatenation: ", df_merged.shape)

            # for all rows in df_merged where the ESM_timestamp is NaN, fill it with the ESM_timestamp_merged
            ## Note: this is necessary because the ESM_timestamp is NaN for all rows which were not merged but merely concatenated
            df_merged = df_merged.reset_index(drop=True)
            for index, row in df_merged.iterrows():
                if pd.isna(row["ESM_timestamp"]):
                    df_merged.loc[index, "ESM_timestamp"] = row["ESM_timestamp_merged"]
            df_merged = df_merged.drop(columns=['ESM_timestamp_merged'])
            print("df_merged.shape after filling ESM_timestamp: ", df_merged.shape)

            # concatenate df_merged to df_final
            df_final = pd.concat([df_final, df_merged], axis=0)
            print("df_final.shape after concatenation: ", df_final.shape)

            # implement going in several steps since its crashing otherwise
            ## save intermediate df_final to csv
            if path_intermediate_files != None:
                #save with pickle highest protocol
                with open(path_intermediate_files + "df_final_" + str(user_count) + ".pkl", 'wb') as f:
                    pickle.dump(df_final, f, pickle.HIGHEST_PROTOCOL)

            #print("Time for User " + str(user_count) + "/" + str(total_users) + " was " + str((time.time()-time_start)/60) + " minutes")
            user_count += 1

        # if sensor timeseries are merged: add prefix to "timestamp_merged" column
        #if add_prefix_to_merged_columns == True:
        #    df_final = df_final.rename(columns={"timestamp_merged": sensor_merge[:3] + "_timestamp_merged"})
        df_final = df_final.rename(columns={"timestamp_merged": sensor_merge[:3] + "_timestamp_merged"})

        return df_final

    # delete rows with missing values
    def impute_deleteNaN(df_final):
        # delete rows with NaN values
        df_final = df_final.dropna(axis=0, how='any')
        return df_final

    #TODO include here imputation methods:
    # - fill by last / next value
    # - fill by mean

high_freq_sensor_sets = [["linear_accelerometer", "gyroscope", "magnetometer", "rotation", "accelerometer"]]
feature_segments = [1] #in seconds
path_intermediate_files = "/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_preparation/features/temp/" #this location is for the intermediate df_final files which will be saved after each user is done. If the


#region merge high-frequency features with speed & acceleration features (or, for feature_segment == 1: merge high-frequency features with speed & acceleration)
## WHAT STILL MISSING: for set "LinAcc-Rot": everything; for set "all highfreq": everything
### NOTE: for LinAcc-Rot set, it crashed always; therefore I implemented a "chunk" approach: after data for one user is merged,
### the intermediate result is stored in "path_intermediate_files". If the function is called again, it starts only after this user
#high_freq_sensor_sets = [["linear_accelerometer", "rotation"],
#                         ["linear_accelerometer", "gyroscope", "magnetometer", "rotation", "accelerometer"]]
#feature_segments = [10,5,2,1] #in seconds
only_active_smartphone_sessions = "yes"
min_gps_accuracy = 35
columns_to_delete = ["device_id", "ESM_timestamp"]
path_features = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/"
path_storage = "/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_preparation/features/"
#path_intermediate_files = "/Users/benediktjordan/Documents/MTS/Iteration01/human_motion/data_preparation/features/temp/" #this location is for the intermediate df_final files which will be saved after each user is done. If the
# script is stopped, the intermediate files can be loaded and the script can be continued with the next user

for high_freq_sensor_set in high_freq_sensor_sets:
    for feature_segment in feature_segments:
        timedelta = str(feature_segment) + "s"
        print("Start with high_freq_sensor_set: " + str(high_freq_sensor_set) + " and feature_segment: " + str(feature_segment))
        df_features_highfreq = pd.read_pickle(path_features + str(high_freq_sensor_set) + "_timeperiod-" + str(
            feature_segment) + "_only-active-smartphone-sessions-" + only_active_smartphone_sessions + "_FeaturesExtracted.pkl")

        # if feature_segment == 1, use speed & acceleration data, as there are no derived features
        if feature_segment == 1:
            df_features_speedacc = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/GPS/locations-aroundevents_features-distance-speed-acceleration_accuracy-less-than" + str(min_gps_accuracy) + "_only-active-smartphone-sessions-" + only_active_smartphone_sessions  + ".csv")
        else:
            df_features_speedacc = pd.read_pickle(path_features + "GPS_timeperiod-" + str(feature_segment) + "_min-gps-accuracy-" + str(min_gps_accuracy) + "_only-active-smartphone-sessions-" + only_active_smartphone_sessions  + "_FeaturesExtracted.pkl")

        df_merged = Merge_and_Impute.merge(df_features_highfreq, df_features_speedacc, "GPS", timedelta, columns_to_delete, path_intermediate_files,
                  add_prefix_to_merged_columns=False)

        #double check if merging worked
        df_merged_timecolumns = df_merged[["timestamp", "GPS_timestamp_merged"]]

        # save with open and pickle highest protocall
        with open(path_storage + str(high_freq_sensor_set) + "-merged-to-GPSFeatures_feature-segment-" + str(feature_segment) + "_min-gps-accuracy-" + str(min_gps_accuracy) +"_only-active-smartphone-sessions-" + only_active_smartphone_sessions + "_FeaturesExtracted_Merged.pkl", 'wb') as f:
            pickle.dump(df_merged, f, pickle.HIGHEST_PROTOCOL)

#endregion