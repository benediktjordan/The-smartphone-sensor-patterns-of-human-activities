#region import
import pandas as pd
import tsfresh
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
import os
import numpy as np
import time
from tqdm import tqdm
import pickle

class computeFeatures:
    # Class Variable
    animal = 'dog'

    # The init method or constructor
    def __init__(self, breed, color):
        # Instance Variable
        self.breed = breed
        self.color = color


    # for battery_charges
    ## calculate difference between "before bed" event/"after bed" event timestamps and "charge battery" event timestamp/"charge
    ## battery double_end_timestamp" timestamp

    def battery_charges_features(self, df):
        # calculate difference between "before bed" event/"after bed" event timestamps and "charge battery" event timestamp/"charge
        # battery double_end_timestamp" timestamp

        return df

#region feature extraction event-wise
def feature_extraction_eventwise(df_sensor, dict_label, sensor_column_names, segment_size, label_column_name, path_intermediate_results, last_esm_timestamp=None, time_column_name = "timestamp", ESM_event_column_name = "ESM_timestamp"):

    df_final = pd.DataFrame()
    df_some_sensors_only = pd.DataFrame()
    counter = 1

    # create list of columns which are used for feature extraction
    sensor_column_names.append(time_column_name)
    sensor_column_names.append("ID")

    # create list of events
    list_of_events = df[ESM_event_column_name].unique().tolist()
    list_of_events.sort()

    # if last_esm_timestamp is not None: get index of last esm timestamp in df.unique() and delete everything before that
    if last_esm_timestamp is not None:
        last_esm_timestamp_index = list_of_events.index(last_esm_timestamp)
        # iterate through all events which are after last_esm_timestamp
        list_of_events = list_of_events[last_esm_timestamp_index+1:]
        #df_final = pd.read_csv(path_intermediate_results + "INTERMEDIATE_df_final_" +str(last_esm_timestamp) + ".csv")
        with open(path_intermediate_results + "INTERMEDIATE_df_final_" +str(last_esm_timestamp) + ".pkl", 'rb') as f:
            df_final = pickle.load(f)

    # add label column to df_sensor (for feature selection)
    df_sensor = labeling_sensor_df(df_sensor, dict_label, label_column_name)

    # iterate through ESM events
    for event in list_of_events:
        # print the portion of events that have been processed
        print(f"Processing {counter} of {len(list_of_events)} events")

        # select data for event
        df_event = df[df[ESM_event_column_name] == event]

        # reset index
        df_event_t = df_event.reset_index(drop=True)

        # create ID column for tsfresh (ID column contains the segment number)
        df_event_t['ID'] = df_event_t.index // segment_size

        # sort by timestamp
        df_event_t = df_event_t.sort_values(by=time_column_name)

        # get only sensor columns
        df_event_t = df_event_t[sensor_column_names]

        # count number of rows with nan value
        nan_rows = df_event_t.isnull().sum(axis=1)
        nan_rows = nan_rows[nan_rows > 0]
        print(f"Number of rows with nan values: {len(nan_rows)}")

        # remove rows with NaN values
        df_event_t = df_event_t.dropna()

        # check if there are any rows; otherwise add to error-dataframe
        if len(df_event_t) == 0:
            if len(df_event) != 0:
                df_some_sensors_only = df_some_sensors_only.append(df_event)
                print("Event with following timestamp has some NaN sensors: " + str(event))
            else:
                df_some_sensors_only = df_some_sensors_only.append(df_event)
                print("Event with following timestamp has no values at all: " + str(event))
            continue

        # check if all segments have <1 elements; if not, delete them
        # explanation: this is necessary for tsfresh; otherwise it will throw an error
        df_event_t = df_event_t.groupby('ID').filter(lambda x: len(x) > 1)

        #extract features
        extracted_features = extract_features(df_event_t, column_id='ID', column_sort=time_column_name)

        # impute missing values
        #impute(extracted_features)

        # apply feature selection
        # create label pd.Series from df_event[label_column_name].iloc[0] with length of extracted_features
        label = pd.Series([df_event[label_column_name].iloc[0]] * len(extracted_features))
        features_filtered = select_features(extracted_features, df_event[label_column_name])

        # add ESM event columns (all ESM event columns)
        extracted_features["ESM_timestamp"] = df_event["ESM_timestamp"].iloc[0]
        #extracted_features["ESM_location"] = df_event["ESM_location"].iloc[0]
        #extracted_features["ESM_location_time"] = df_event["ESM_location_time"].iloc[0]
        #extracted_features["ESM_bodyposition"] = df_event["ESM_bodyposition"].iloc[0]
        #extracted_features["ESM_bodyposition_time"] = df_event["ESM_bodyposition_time"].iloc[0]
        #extracted_features["ESM_activity"] = df_event["ESM_activity"].iloc[0]
        #extracted_features["ESM_activity_time"] = df_event["ESM_activity_time"].iloc[0]
        #extracted_features["ESM_smartphonelocation"] = df_event["ESM_smartphonelocation"].iloc[0]
        #extracted_features["ESM_smartphonelocation_time"] = df_event["ESM_smartphonelocation_time"].iloc[0]
        #extracted_features["ESM_aligned"] = df_event["ESM_aligned"].iloc[0]
        #extracted_features["ESM_aligned_time"] = df_event["ESM_aligned_time"].iloc[0]
        #extracted_features["User ID"] = df_event["2"].iloc[0]

        # add to final dataframe
        df_final = df_final.append(extracted_features)

        # save intermediate results for every 10 events
        if counter % 10 == 0:
            with open(path_intermediate_results + "INTERMEDIATE_df_final_" + str(event) + '.pkl', 'wb') as f:
                pickle.dump(df_final, f, pickle.HIGHEST_PROTOCOL)
        print("Event with following timestamp has been processed: " + str(event))
        counter += 1

    #reset index
    df_final = df_final.reset_index(drop=True)
    return df_final, df_some_sensors_only

#intermediate: extract features for merged high-frequency sensors
# TODO: extract features also for low-frequency sensors (before, they have to be merged, off course)

merged_sensors = ["accelerometer", "gravity", "gyroscope", "linear_accelerometer", "magnetometer", "rotation"]
# get sensor_columns for merged sensors
sensor_columns_merged_sensors = []
for sensor in merged_sensors:
    sensor_columns_merged_sensors.append(sensor_columns[sensor])
# combine list elements into one element
sensor_columns_merged_sensors = [item for sublist in sensor_columns_merged_sensors for item in sublist]

#load data
path_merged_sensorfile = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/merged/all_esm_timeperiod_5 min_TimeseriesMerged.csv"
df_merged = pd.read_csv(path_merged_sensorfile)

# extract features for different time periods

# complete loop
time_periods = [2, 1, 5, 10] #in seconds
frequency = 10 #in Hz
dir_sensorfiles = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/"
path_intermediate_results = dir_sensorfiles + "data_preparation/features/intermediate/"

for seconds in tqdm(time_periods):
    time.start = time.time()
    segment_size = seconds * frequency
    df_features, df_some_sensors_only = feature_extraction_eventwise(df_merged, sensor_columns_merged_sensors, segment_size, "timestamp", "ESM_timestamp", path_intermediate_results = path_intermediate_results)

    # save features
    df_features.to_csv(dir_sensorfiles + "features/" + str(seconds) + " seconds_high_frequency_sensors.csv")
    df_some_sensors_only.to_csv(dir_sensorfiles + "features/" + str(seconds) + " seconds_high_frequency_sensors_ErrorCases.csv")
    print("Time for " + str(seconds) + " seconds: " + str(time.time() - time.start))




# loop if there has been a break before
time_periods = [2, 1, 5, 10] #in seconds
frequency = 10 #in Hz
dir_sensorfiles = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/"
path_intermediate_results = dir_sensorfiles + "/data_preparation/features/intermediate/"
last_esm_timestamp = "2022-06-23 07:15:28.816999936"

for seconds in time_periods:
    time.start = time.time()
    segment_size = seconds * frequency
    df_features, df_some_sensors_only = feature_extraction_eventwise(df_merged, sensor_columns_merged_sensors, segment_size, "timestamp", "ESM_timestamp", path_intermediate_results = path_intermediate_results,
                                                           last_esm_timestamp = last_esm_timestamp)

    # save features
    df_features.to_csv(dir_sensorfiles + "features/" + str(seconds) + " seconds_high_frequency_sensors.csv")
    df_some_sensors_only.to_csv(dir_sensorfiles + "data_preparation/features/" + str(seconds) + " seconds_high_frequency_sensors_ErrorCases.csv")
    print("Time for " + str(seconds) + " seconds: " + str(time.time() - time.start))

#endregion

#region feature extraction (not element-wise)

#testcase
df = df_merged
sensor_column_names = sensor_columns_merged_sensors
segment_size = 20
timestamp_column = "timestamp"
esm_timestamp_column = "ESM_timestamp"


#function to extract and select features based on the tsfresh algorithms
# note: label-column is needed since for tsfresh feature selection
def feature_extraction(df, sensor_column_names, segment_size,  time_column_name = "timestamp", ESM_event_column_name = "ESM_timestamp"):

    # reset index
    # TODO: check if this is necessary and maybe event dangerous; is df_merged ordered by event and timestamp?
    df = df.reset_index(drop=True)

    # create ID column for tsfresh (ID column contains the segment number)
    df['ID'] = df.index // segment_size
    #sensor_column_names.append(time_column_name)
    sensor_column_names.append("ID")

    # get only sensor columns
    df = df[sensor_column_names]

    # count number of rows with nan value
    nan_rows = df.isnull().sum(axis=1)
    nan_rows = nan_rows > 0
    print(f"Number of rows with nan values: {len(nan_rows[nan_rows == True])}")

    # remove rows with NaN values
    df = df.dropna()
    df = df.reset_index(drop=True)

    # check if all segments have <1 elements; if not, delete them
    df = df.groupby('ID').filter(lambda x: len(x) > 1)

    #extract features
    extracted_features = extract_features(df, column_id='ID')

    # save intermediate result
    extracted_features.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/features/intermediate/extracted_features.csv")

    # impute missing values
    #impute(extracted_features)

    # apply feature selection
    #features_filtered = select_features(extracted_features, y)

    # add ESM event columns (all ESM event columns)
    df_esms = df.iloc[::segment_size, :]
    extracted_features["ESM_timestamp"] = df_esms["ESM_timestamp"]
    extracted_features["ESM_location"] = df_esms["ESM_location"]
    extracted_features["ESM_location_time"] = df_esms["ESM_location_time"]
    extracted_features["ESM_bodyposition"] = df_esms["ESM_bodyposition"]
    extracted_features["ESM_bodyposition_time"] = df_esms["ESM_bodyposition_time"]
    extracted_features["ESM_activity"] = df_esms["ESM_activity"]
    extracted_features["ESM_activity_time"] = df_esms["ESM_activity_time"]
    extracted_features["ESM_smartphonelocation"] = df_esms["ESM_smartphonelocation"]
    extracted_features["ESM_smartphonelocation_time"] = df_esms["ESM_smartphonelocation_time"]
    extracted_features["ESM_aligned"] = df_esms["ESM_aligned"].iloc[0]
    extracted_features["ESM_aligned_time"] = df_esms["ESM_aligned_time"]
    extracted_features["User ID"] = df_esms["2"]

    return extracted_features


# create list of sensor column names
merged_sensors = ["accelerometer", "gravity", "gyroscope", "linear_accelerometer", "magnetometer", "rotation"]
# get sensor_columns for merged sensors
sensor_columns_merged_sensors = []
for sensor in merged_sensors:
    sensor_columns_merged_sensors.append(sensor_columns[sensor])
# combine list elements into one element
sensor_columns_merged_sensors = [item for sublist in sensor_columns_merged_sensors for item in sublist]

#load data
path_merged_sensorfile = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/merged/all_esm_timeperiod_5 min_TimeseriesMerged.csv"
df_merged = pd.read_csv(path_merged_sensorfile, nrows = 100000)

# extract features for different time periods

# complete loop
time_periods = [2, 1, 5, 10] #in seconds
frequency = 10 #in Hz
dir_sensorfiles = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/"

for seconds in tqdm(time_periods):
    time.start = time.time()
    segment_size = seconds * frequency
    df_features = feature_extraction(df_merged, sensor_columns_merged_sensors, segment_size, "timestamp", "ESM_timestamp")

    # save features
    df_features.to_csv(dir_sensorfiles + "data_preparation/features/" + str(seconds) + " seconds_high_frequency_sensors_notelementwise.csv")
    print("Time for " + str(seconds) + " seconds: " + str(time.time() - time.start))

#endregion




#region testcase
df = pd.read_csv(
    "/Users/benediktjordan/Documents/MTS/Iteration01/Data/features/merged/all_esm_timeperiod_5 min_TimeseriesMerged.csv",
    parse_dates=['timestamp'], infer_datetime_format=True, nrows= 10000)
sensor_column_names = ['acc_double_values_0', 'acc_double_values_1', 'acc_double_values_2']
segment_size = 100
time_column_name = 'timestamp'
ESM_event_column_name = 'ESM_timestamp'
df_final = feature_extraction(df, sensor_column_names, segment_size, time_column_name, ESM_event_column_name)

if __name__ == '__main__':
    # create object of class
    obj = computeFeatures('German Shepherd', 'Black')

    # call the instance method using the object obj
    print(obj.battery_charges_features('df'))

    # call the class method using the class name
    print(computeFeatures.animal)

    # call the static method using the class name
    print(computeFeatures.battery_charges_features('df'))

    # call the feature extraction function
    print(feature_extraction('df', 'sensor_column_names', 'segment_size', 'time_column_name', 'ESM_event_column_name'))
#impute missing values
extracted_features = extracted_features.fillna(extracted_features.mean())
return extracted_features


#region load testing data
df = pd.read_csv(
    "/Users/benediktjordan/Documents/MTS/Iteration01/Data/accelerometer_esm_timeperiod_5 min.csv_JSONconverted.csv",
    parse_dates=['timestamp'], infer_datetime_format=True, nrows= 10000)

# select only necessary columns
df = df[[ 'timestamp', "acc_double_values_0"]]
# remove rows with NaN values
df = df.dropna(how='any')
# create id column which creates a new id for every 200 rows
df['id'] = df.index // 200

extracted_features = extract_features(df, column_id="id", column_sort="timestamp")

# remove NaN features & select only relevant features
# TODO: find out how "relevant" is defined?
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute

impute(extracted_features)
features_filtered = select_features(extracted_features, y)

# IMPORTANT: tsfresh seemed once to not  work with apple silicon; use rosetta 2 to run this code; but finally it now works with Apple Silicon
# here a guide how to create x86 environment: https://towardsdatascience.com/how-to-manage-conda-environments-on-an-apple-silicon-m1-mac-1e29cb3bad12

from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, \
    load_robot_execution_failures
download_robot_execution_failures()
timeseries, y = load_robot_execution_failures()
#endregion
#endregion

#region TSFEL: alternative library for feature extraction
import tsfel
import zipfile
import numpy as np
import pandas as pd
import wget
import os

# Load the dataset from online repository
wget.download('https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip', '/Users/benediktjordan/Downloads/test.zip')

# Unzip the dataset
zip_ref = zipfile.ZipFile("/Users/benediktjordan/Downloads/UCI HAR Dataset.zip", 'r')
zip_ref.extractall()
zip_ref.close()

# Store the dataset as a Pandas dataframe.
x_train_sig = np.loadtxt('UCI HAR Dataset/train/Inertial Signals/total_acc_x_train.txt', dtype='float32')
X_train_sig = pd.DataFrame(np.hstack(x_train_sig), columns=["total_acc_x"])

X_train_sig.head()

cfg_file = tsfel.get_features_by_domain()                                                        # If no argument is passed retrieves all available features
X_train = tsfel.time_series_features_extractor(cfg_file, X_train_sig, fs=50, window_size=250)

#endregion