#region import
# pip install mysql-connector-python
import mysql.connector
import pandas as pd

import os
import time
import json

import sqlite3
from pandas import read_sql_query, read_sql_table

#endregion



# region join and edit the CSVs which are downloaded from AWS S3 storage (instead of directly from MySQL database)
## join csv
base_path = "/Volumes/TOSHIBA EXT/InUsage/Databases/db_iteration1_20220628/gravity from S3/"
file_list = ["LOAD00000001.csv",
                "LOAD00000002.csv",
                "LOAD00000003.csv",
                "LOAD00000004.csv"]
file = file_list[0]
df_to_add = pd.read_csv(base_path+file_list[3])
df_all = pd.DataFrame()
#df_all = pd.concat([df_all, df_to_add])
df_all = pd.DataFrame( np.concatenate( (df_all.values, df_to_add.values), axis=0 ) )


def join_edit_csv(base_path, file_list):
    df_all = pd.DataFrame()
    #join files to one dataframe
    for file in file_list:
        df_to_add = pd.read_csv(base_path+file, header=None)
        df_all = pd.concat([df_all, df_to_add], ignore_index=True)
        print("Finished adding file: "+ str(file))

    #df_all = df_all.reset_index(drop=False)
    return df_all

dir_databases = "/Volumes/INTENSO/In Usage/db_iteration1_20220717"
for root, dirs, files in os.walk(dir_databases):  # iterate through different subfolders
    for sensor in dirs:
#        if os.path.exists(dir_databases + "/" + subfolder):
#            sensor_list = os.listdir(dir_databases + "/" + subfolder)
#            for sensor in sensor_list:
        time_begin = time.time()
        #file_list = os.listdir(dir_databases +  "/" + sensor) # for Windows
        file_list = [i for i in os.listdir(dir_databases +  "/" + sensor) if not i.startswith(".")] #for Mac

        df_all = join_edit_csv(dir_databases +  "/" + sensor + "/", file_list)
        df_all.to_csv(dir_databases + "/" + sensor  + ".csv")
        time_end = time.time()
        print("sensor " + sensor + " took " + str(time_end-time_begin) + " seconds")

#check if correct
df_all = pd.read_csv("/Volumes/INTENSO/In Usage/Databases/db_iteration1_20220712/gyroscope.csv", nrows=10)

#endregion

#region transform SQLite files which have been directly downloaded from iPhones of probands
## create transformation function
def transform_sqlite(path_sqlite, sensor_name, sensor_tablename):

    #connect to database
    with sqlite3.connect(path_sqlite) as dbcon:
        tables = list(read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", dbcon)['name'])
        dict_sensor = {tbl : read_sql_query(f"SELECT * from {tbl}", dbcon) for tbl in tables}

    df_sensor = dict_sensor[sensor_tablename]

    #rename columns
    if sensor_name == "Rotation":
        counter = 1
        rows = []
        for index, row in df_sensor.iterrows():
            selection = pd.Series([row["ZLABEL"], row["ZACCURACY"], row["ZDEVICE_ID"], row["ZTIMESTAMP"],
                                   row["ZDOUBLE_VALUES_0"], row["ZDOUBLE_VALUES_1"], row["ZDOUBLE_VALUES_2"],
                                   row["ZDOUBLE_VALUES_3"]],
                                  index=["label", "accuracy", "device_id", "timestamp", "double_values_0",
                                         "double_values_1", "double_values_2", "double_values_3"])
            selection_json = selection.to_json()
            row = [counter, row["ZTIMESTAMP"], row["ZDEVICE_ID"], selection_json]
            rows.append(row)
            counter += 1
            print("Progress = " + str(counter) + "/" + str(len(df_sensor)))

        df = pd.DataFrame(rows, columns=[0, 1, 2, 3])

    else:
        counter = 1
        rows = []
        for index, row in df_sensor.iterrows():
            selection = pd.Series([row["ZLABEL"], row["ZACCURACY"], row["ZDEVICE_ID"], row["ZTIMESTAMP"],
                                   row["ZDOUBLE_VALUES_0"], row["ZDOUBLE_VALUES_1"], row["ZDOUBLE_VALUES_2"]],
                                  index = ["label", "accuracy", "device_id", "timestamp", "double_values_0",
                                           "double_values_1","double_values_2" ])
            selection_json = selection.to_json()
            row = [counter, row["ZTIMESTAMP"], row["ZDEVICE_ID"], selection_json]
            rows.append(row)
            counter += 1
            print("Progress = " + str(counter) + "/" + str(len(df_sensor)))

        df = pd.DataFrame(rows, columns = [0,1,2,3])

    return df

## iterate through all the files
paths_sqlite = ["/Users/benediktjordan/Documents/In Usage/Databases/db_fromphonesdirectly/AWARE_Gyroscope_Bene.sqlite",
                "/Users/benediktjordan/Documents/In Usage/Databases/db_fromphonesdirectly/AWARE_LinearAccelerometer_Bene.sqlite",
                "/Users/benediktjordan/Documents/In Usage/Databases/db_fromphonesdirectly/AWARE_Rotation_Tina.sqlite",
                "/Users/benediktjordan/Documents/In Usage/Databases/db_fromphonesdirectly/AWARE_Rotation_Selcuk.sqlite",
                "/Users/benediktjordan/Documents/In Usage/Databases/db_fromphonesdirectly/AWARE_Rotation_Bene.sqlite",
                ]
sensor_names = ["Gyroscope",
                "Linear Accelerometer",
                "Rotation",
                "Rotation",
                "Rotation"]
sensor_tablenames = ["ZAWAREGYROSCOPEOM",
                     "ZAWARELINEARACCELEROMETEROM",
                     "ZAWAREROTATIONOM","ZAWAREROTATIONOM",
                     "ZAWAREROTATIONOM"]


for i in range(len(paths_sqlite)):
    path_sqlite = paths_sqlite[i]
    sensor_name = sensor_names[i]
    sensor_tablename = sensor_tablenames[i]
    df = transform_sqlite(path_sqlite, sensor_name, sensor_tablename)
    df.to_csv("/Users/benediktjordan/Documents/In Usage/Databases/db_fromphonesdirectly/" + path_sqlite + ".csv")






db_path = "/Users/benediktjordan/Documents/In Usage/Databases/db_fromphonesdirectly/"
sensor_name = "Gyroscope"
sensor_tablename = "ZAWAREGYROSCOPEOM"
path_sqlite = db_path + "AWARE_Gyroscope_Bene.sqlite"

sensor_name = "Linear Accelerometer"
sensor_tablename = "ZAWARELINEARACCELEROMETEROM"
path_sqlite = db_path + "AWARE_LinearAccelerometer_Bene.sqlite"


sensor_name = "Rotation"
sensor_tablename = "ZAWAREROTATIONOM"
path_sqlite = db_path + "AWARE_Rotation_Tina.sqlite"

sensor_name = "Rotation"
sensor_tablename = "ZAWAREROTATIONOM"
path_sqlite = db_path + "AWARE_Rotation_Selcuk.sqlite"
df = transform_sqlite(path_sqlite,sensor_name, sensor_tablename)
df.to_csv(db_path + sensor_name + "_Selcuk.csv")

sensor_name = "Rotation"
sensor_tablename = "ZAWAREROTATIONOM"
path_sqlite = db_path + "AWARE_Rotation_Bene.sqlite"
df = transform_sqlite(path_sqlite,sensor_name, sensor_tablename)
df.to_csv(db_path + sensor_name + "_Bene.csv")



df = transform_sqlite(path_sqlite,sensor_name, sensor_tablename)
df.to_csv(db_path + sensor_name + "_Tina.csv")

#endregion

#region change column names to 1,2,3 of sensor files (necessary for the ones downloaded from S3 and I didn´t do it directly)
dir_databases = "/Volumes/INTENSO/In Usage new/Databases"
for root, dirs, files in os.walk(dir_databases):  # iterate through different subfolders
    for subfolder in dirs:
        for index, entry in sensors_and_frequencies.iterrows():
            sensor = entry["Sensor"]
            path_sensor = dir_databases + "/" + subfolder + "/" + sensor + ".csv"
            if os.path.exists(path_sensor):
                df_all = pd.read_csv(path_sensor, nrows=10)
                #print("we are in " + path_sensor)
                try:
                    df_all["2"]
                    if len(df_all.columns) > 5:
                        #df_all = pd.read_csv(path_sensor)
                        #df_all = df_all.iloc[:, 0:4]
                        #df_all.to_csv(path_sensor)
                        print("More then 4 columns in: " + path_sensor)
                except:
                    #df_all = pd.read_csv(path_sensor)
                    #df_all = df_all.iloc[:, 0:4]
                    #df_all.rename(columns={df_all.columns[1]: '1'}, inplace=True)
                    #df_all.rename(columns={df_all.columns[2]: '2'}, inplace=True)
                    #df_all.rename(columns={df_all.columns[3]: '3'}, inplace=True)
                    #df_all.to_csv(path_sensor)
                    print("Wrong column names in: " + path_sensor)

toomany_columns = ["/Volumes/INTENSO/In Usage new/Databases/db_iteration1_20220717/linear_accelerometer.csv",
                   "/Volumes/INTENSO/In Usage new/Databases/db_iteration1_20220723/accelerometer.csv",
                   "/Volumes/INTENSO/In Usage new/Databases/db_iteration1_20220723/gravity.csv",
                     "/Volumes/INTENSO/In Usage new/Databases/db_iteration1_20220723/gyroscope.csv",
                        "/Volumes/INTENSO/In Usage new/Databases/db_iteration1_20220723/linear_accelerometer.csv",
                        "/Volumes/INTENSO/In Usage new/Databases/db_iteration1_20220723/locations.csv",
                        "/Volumes/INTENSO/In Usage new/Databases/db_iteration1_20220723/magnetometer.csv",
                        "/Volumes/INTENSO/In Usage new/Databases/db_iteration1_20220723/rotation.csv",
                   "/Volumes/INTENSO/In Usage new/Databases/db_iteration1_20220717/gyroscope.csv",
                   "/Volumes/INTENSO/In Usage new/Databases/db_iteration1_20220712/gyroscope.csv"]


df_all = pd.read_csv(toomany_columns[9], nrows=10)

time.sleep(3200)
time_begin  = time.time()
df_all = pd.read_csv(toomany_columns[9])
df_all = df_all.iloc[:, 1:5]
df_all.rename(columns={df_all.columns[0]: '0'}, inplace=True)
df_all.rename(columns={df_all.columns[1]: '1'}, inplace=True)
df_all.rename(columns={df_all.columns[2]: '2'}, inplace=True)
df_all.rename(columns={df_all.columns[3]: '3'}, inplace=True)

df_all.to_csv(toomany_columns[9])
time_end = time.time()
print(time_end - time_begin)
# test if it worked
df_all = pd.read_csv(toomany_columns[9], nrows=10)

df_all2 = pd.read_csv(toomany_columns[4])


#endregion

# region manipulate ESM
# region labels: join all ESM files

#endregion

#region create ESM file with one row per ESM_timestamp and one column per ESM_question
def esm_transform(path_esm_all):

    ## Read the JSON parts of the ESM
    esm_all = pd.read_csv(path_esm_all)
    esm_all = esm_all.reset_index(drop=True)
    stdf = esm_all["3"].apply(json.loads)
    esm_all_transformed = pd.DataFrame(stdf.tolist()) # or stdf.apply(pd.Series)
    esm_all_transformed = pd.concat([esm_all, esm_all_transformed], axis=1)
    esm_all_transformed = esm_all_transformed.dropna()#there is 1 NAN value in the ESMs

    ## Create a new dataframe with one row per ESM_timestamp and one column per ESM_question
    esm_all_final = pd.DataFrame(columns=["timestamp", "device_id", "location", "location_time",
                                          "bodyposition", "bodyposition_time", "activity", "activity_time",
                                          "smartphonelocation", "smartphonelocation_time", "aligned", "aligned_time"])
    timestamp = []
    device_id = []
    location = []
    location_time = []
    bodyposition = []
    bodyposition_time = []
    activity = []
    activity_time = []
    smartphonelocation = []
    smartphonelocation_time = []
    aligned = []
    aligned_time = []
    double_entries = []

    for esm_entry in esm_all_transformed["1"].unique(): #iterate through all unique ESM events
        esm_entry_subset = esm_all_transformed[esm_all_transformed["1"] == esm_entry]
        timestamp.append(esm_entry)
        device_id.append(esm_entry_subset["2"].unique()[0])

        location_filled = 0
        bodyposition_filled = 0
        activity_filled = 0
        smartphonelocation_filled = 0
        aligned_filled = 0
        for index, entry in esm_entry_subset.iterrows(): # iterate through all questions in every unique ESM event

            if "Current Location" in entry["esm_json"] and location_filled == 0: # IMPORTANT: sometimes answers are reported several times (technical bug? if somebody hits a button twice?) -> only take the first answer
                location.append(entry["esm_user_answer"])
                location_time.append(entry["double_esm_user_answer_timestamp"])
                location_filled += 1

            elif "Body Motion" in entry["esm_json"] and bodyposition_filled == 0:
                bodyposition.append(entry["esm_user_answer"])
                bodyposition_time.append(entry["double_esm_user_answer_timestamp"])
                bodyposition_filled += 1

            elif ("Activity" in entry["esm_json"] or "Physical Activity" in entry["esm_json"]) and activity_filled == 0:
                activity.append(entry["esm_user_answer"])
                activity_time.append(entry["double_esm_user_answer_timestamp"])
                activity_filled += 1

            elif "Phone Location" in entry["esm_json"] and smartphonelocation_filled == 0:
                smartphonelocation.append(entry["esm_user_answer"])
                smartphonelocation_time.append(entry["double_esm_user_answer_timestamp"])
                smartphonelocation_filled += 1

            elif "Evaluation of Phone Use" in entry["esm_json"] and aligned_filled == 0:
                aligned.append(entry["esm_user_answer"])
                aligned_time.append(entry["double_esm_user_answer_timestamp"])
                aligned_filled += 1

            else:
                double_entries.append(index)

        if location_filled >1:
            print("There is a problem with the location. The ESM timestamp is: " + str(esm_entry) + " Indices are: " + str(esm_entry_subset.index))
        if bodyposition_filled >1:
            print("There is a problem with the bodyposition. The ESM timestamp is: " + str(esm_entry) + " Indices are: " + str(esm_entry_subset.index))
        if activity_filled >1:
            print("There is a problem with the activity. The ESM timestamp is: " + str(esm_entry) + " Indices are: " + str(esm_entry_subset.index))
        if smartphonelocation_filled >1:
            print("There is a problem with the smartphonelocation. The ESM timestamp is: " + str(esm_entry) + " Indices are: " + str(esm_entry_subset.index))
        if aligned_filled >1:
            print("There is a problem with the aligned. The ESM timestamp is: " + str(esm_entry) + " Indices are: " + str(esm_entry_subset.index))

        #check if all questions have been filled; if not, fill with NaN
        if location_filled == 0:
            location.append(np.nan)
            location_time.append(np.nan)
        if bodyposition_filled == 0:
            bodyposition.append(np.nan)
            bodyposition_time.append(np.nan)
        if activity_filled == 0:
            activity.append(np.nan)
            activity_time.append(np.nan)
        if smartphonelocation_filled == 0:
            smartphonelocation.append(np.nan)
            smartphonelocation_time.append(np.nan)
        if aligned_filled == 0:
            aligned.append(np.nan)
            aligned_time.append(np.nan)

    # create the final dataframe
    esm_all_final["timestamp"] = timestamp
    esm_all_final["device_id"] = device_id
    esm_all_final["location"] = location
    esm_all_final["location_time"] = location_time
    esm_all_final["bodyposition"] = bodyposition
    esm_all_final["bodyposition_time"] = bodyposition_time
    esm_all_final["activity"] = activity
    esm_all_final["activity_time"] = activity_time
    esm_all_final["smartphonelocation"] = smartphonelocation
    esm_all_final["smartphonelocation_time"] = smartphonelocation_time
    esm_all_final["aligned"] = aligned
    esm_all_final["aligned_time"] = aligned_time

    # remove the "[ ]" from the ESM colums
    esm_all_final["location"] = esm_all_final["location"].str[2:]
    esm_all_final["location"] = esm_all_final["location"].str[:-2]
    esm_all_final['bodyposition'] = esm_all_final['bodyposition'].str[2:]
    esm_all_final['bodyposition'] = esm_all_final['bodyposition'].str[:-2]
    esm_all_final['activity'] = esm_all_final['activity'].str[2:]
    esm_all_final['activity'] = esm_all_final['activity'].str[:-2]
    esm_all_final['smartphonelocation'] = esm_all_final['smartphonelocation'].str[2:]
    esm_all_final['smartphonelocation'] = esm_all_final['smartphonelocation'].str[:-2]
    esm_all_final['aligned'] = esm_all_final['aligned'].str[2:]
    esm_all_final['aligned'] = esm_all_final['aligned'].str[:-2]


    return esm_all_final

#esm_all_final.isnull().sum() #count NaN values
path_esm_all = "/Volumes/INTENSO/In Usage new/Databases/esm_all.csv"
esm_all_final = esm_transform(path_esm_all)
esm_all_final.to_csv(dir_databases + "/esm_all_transformed.csv")

#endregion



# count numbers in ESM file
esm_counts = esm_all_final["esm_user_answer"].value_counts().to_frame('count')

# create ESM file based on events (single ESMs)
## iterate through timstamps and create one row per timestamp
esm_all_final_single = pd.DataFrame()
for timestamp in esm_all_final["timestamp"]:
    timestamp_data = esm_all_final[esm_all_final["timestamp"] == timestamp]
    for index, row in timestamp_data.iterrows():



        esm_all_final_single = esm_all_final_single.append(row)
    esm_all_final_single = esm_all_final_single.append(esm_all_final[esm_all_final["timestamp"] == timestamp])
    esm_all_final_single = esm_all_final_single.reset_index(drop=True)
# endregion

#region filter/select only data in x minute range around ES events
# TODO: check why I still have the file "gravity_from_S3" and solve the problem
# TODO: make sure that only ESM_timestamps (as the identifier) remain in sensordata

#TODO before selecting xmin around event, delete events which are less than 1/5 minutes apart from each other (for same participant!)

dir_databases = "/Volumes/INTENSO/In Usage new/Databases"
#sensor = "barometer"
path_esm ="/Volumes/INTENSO/In Usage new/Databases/esm_all_transformed.csv"
time_period = 5


#temporary
path = "/Volumes/INTENSO/In Usage new/Databases/db_iteration1_20220712/locations.csv"
##test speed of reading data
start = time.time()
sensor_all = pd.read_csv(path)
end = time.time()
print("withoug dtype specification " + str(end - start))

##test speed of reading data
start = time.time()
sensor_all = pd.read_csv(path, dtype={"0": 'int64', "1": 'float64', "2": object, "3": object})
end = time.time()
print("with dtype specificationn" + str(end - start))



def filter_sensordata_by_esm(dir_databases, path_esm, sensor, frequency, time_period):
    paths_intermediate = []

    # check for which ESM events only partly sensordata has been found
    esm_partly_sensor = []
    esm_partly_index = []
    esm_partly_percentage = []

    esm_all_transformed = pd.read_csv(path_esm)
    #esm_all_withouduplicates = esm_all.drop_duplicates(subset=['1']) #delete duplicates in esm_all
    time_period_unix = time_period * 60 * 1000 # transform minutes into unix time

    # iterate through every sensor file and find for every ESM timestamp the corresponding sensor data in a [time_period] minute range
    for root, dirs, files in os.walk(dir_databases):  # iterate through different subfolders
        for subfolder in dirs:
            time_start = time.time()
            print("We are in subfolder: " + subfolder)

            #check if I was in this subfolder before
            #path_before = "/Users/benediktjordan/Documents/MTS/Iteration01/Data"  + "/TEMPORARY_" + sensor + "_" + subfolder + "_esm_timeperiod_" + str(time_period) + " min.csv"
            #if os.path.exists(path_before):
            #    paths_intermediate.append(path_before)
            #    print("We have already been in this subfolder before")
            #    continue
            path_sensor = dir_databases + "/" + subfolder + "/" + sensor + ".csv"

            if os.path.exists(path_sensor):
                sensor_all = pd.read_csv(path_sensor, dtype={"0": 'int64', "1": 'float64', "2": object, "3": object})
                df = pd.DataFrame()

                counter_esm = 0

                #iterate through all ESM events & find corresponding sensor data
                for index, esm_entry in esm_all_transformed.iterrows():

                    #find all sensor data in time period around esm_entry filtered by the user
                    sensor_data_in_time_period = sensor_all[(sensor_all["2"] == esm_entry["device_id"]) & (sensor_all["1"] >= esm_entry["timestamp"] - time_period_unix) & (sensor_all["1"] <= esm_entry["timestamp"] + time_period_unix)].copy()

                    #check if empty
                    if len(sensor_data_in_time_period) != 0:
                        #check if data is correct
                        print( sensor + " segment length for ESM index: " + str(index) + ":"+ str(len(sensor_data_in_time_period)) + "/" + str(time_period*2*60*frequency))

                        #add ESM columns to sensor data
                        sensor_data_in_time_period["ESM_timestamp"] = esm_entry["timestamp"]
                        sensor_data_in_time_period["ESM_location"] = esm_entry["location"]
                        sensor_data_in_time_period["ESM_location_time"] = esm_entry["location_time"]
                        sensor_data_in_time_period["ESM_bodyposition"] = esm_entry["bodyposition"]
                        sensor_data_in_time_period["ESM_bodyposition_time"] = esm_entry["bodyposition_time"]
                        sensor_data_in_time_period["ESM_activity"] = esm_entry["activity"]
                        sensor_data_in_time_period["ESM_activity_time"] = esm_entry["activity_time"]
                        sensor_data_in_time_period["ESM_smartphonelocation"] = esm_entry["smartphonelocation"]
                        sensor_data_in_time_period["ESM_smartphonelocation_time"] = esm_entry["smartphonelocation_time"]
                        sensor_data_in_time_period["ESM_aligned"] = esm_entry["aligned"]
                        sensor_data_in_time_period["ESM_aligned_time"] = esm_entry["aligned_time"]

                        #delete this row from ESM file (so it will not be checked again in other sensor files)
                        # but only if a sufficient amount of sensor data has been found; otherwise: leave row so remaining sensor data might be found
                        # in other databases
                        # and only check this for sensors which actually have a frequency
                        if frequency != 0:
                            if (len(sensor_data_in_time_period) / (time_period * 2 * 60 * frequency)) > 0.9:
                                esm_all_transformed = esm_all_transformed.drop(index)
                            else:
                                esm_partly_sensor.append(sensor)
                                esm_partly_index.append(index)
                                esm_partly_percentage.append(len(sensor_data_in_time_period) / (time_period * 2 * 60 * frequency))


                        #print(subfolder + " " + sensor + " " + str(index) + "/" + str(len(esm_all_transformed)))

                        #add sensor data to dataframe
                        df = pd.concat([df, sensor_data_in_time_period], ignore_index=True, sort=False)

                        counter_esm += 1
                    else:
                        #print("I continued")
                        continue

                #save intermediate results (after each database)
                df.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data"  + "/TEMPORARY_" + sensor + "_" + subfolder + "_esm_timeperiod_" + str(time_period) + " min.csv",index=False)
                paths_intermediate.append("/Users/benediktjordan/Documents/MTS/Iteration01/Data"  + "/TEMPORARY_" + sensor + "_" + subfolder + "_esm_timeperiod_" + str(time_period) + " min.csv")

                time_end = time.time()
                print("finished " + sensor + " in " + subfolder + " in " + str(time_end - time_start) + " seconds")
                print("There have been " + str(counter_esm) + " ESM events for sensor "+ sensor + " in this database")
                if frequency !=0:
                    if counter_esm != 0:
                        print("real/planned datapoints = %:" + str(len(df))+ "/" + str(counter_esm*time_period*2*60*frequency) + " = " + str((len(df)/(counter_esm*time_period*2*60*frequency))*100) + "%")

            else:
                print("File does not exist: " + path_sensor)
                continue

    #join the intermediate dataframes
    df_final = pd.DataFrame()
    for path_intermediate in paths_intermediate:
        # check if path_intermediate contains any data; if not -> continue
        try:
            df_intermediate = pd.read_csv(path_intermediate)
            df_final = pd.concat([df_final, df_intermediate], ignore_index=True, sort=False)
            print("finished " + path_intermediate)
        except:
            print("Error for file: " + path_intermediate)
            continue

    # join the lists which contain information about partly sensor data
    sensor_partly_data = pd.DataFrame()
    sensor_partly_data["sensor"] = esm_partly_sensor
    sensor_partly_data["index"] = esm_partly_index
    sensor_partly_data["percentage"] = esm_partly_percentage

    return df_final, sensor_partly_data

#sensors_and_frequencies = sensors_and_frequencies.drop(labels=[0,1,2,3,4,5,6,7,9,11], axis=0)
# drop first rows
for index, entry in sensors_and_frequencies.iterrows():
    sensor = entry["Sensor"]
    frequency = entry["Frequency (in Hz)"]
    df, sensor_partly_data = filter_sensordata_by_esm(dir_databases, path_esm, sensor, frequency, time_period)
    df.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data" + "/FINAL_" + sensor + "_esm_timeperiod_" + str(time_period) + " min.csv",index=False)
    sensor_partly_data.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data" + "/FINAL_SENSORPARTLY_" + sensor + "_esm_timeperiod_" + str(time_period) + " min.csv",index=False)

"
#temporary 4
sensor = "locations"
frequency = 1
df, sensor_partly_data = filter_sensordata_by_esm(dir_databases, path_esm, sensor, frequency, time_period)
df.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data" + "/FINAL_" + sensor + "_esm_timeperiod_" + str(time_period) + " min.csv",index=False)
sensor_partly_data.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data" + "/FINAL_SENSORPARTLY_" + sensor + "_esm_timeperiod_" + str(time_period) + " min.csv",index=False)

## this is to double check the functionality of this function with analytics
df_esm = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled.csv")
# create ESM_timestamp column which is in datetime format from timestamp column
df_esm["ESM_timestamp"] = pd.to_datetime(df_esm["timestamp"], unit = "ms")
df_esm = df_esm[["ESM_timestamp", "device_id"]]

df_sensor = df.copy()
time_period = 90
df_sensor["timestamp"] = pd.to_datetime(df_sensor["1"], unit = "ms")
# iterate through ESM_timestamps in df_esm
df_results = pd.DataFrame()
for index, row in df_esm.iterrows():
    # find data in time_period around ESM_timestamp in df_sensors_all
    event_time = row["ESM_timestamp"]
    df_temporary = df_sensor[(df_sensor['timestamp'] >= event_time - pd.Timedelta(seconds=(time_period / 2))) & (
            df_sensor['timestamp'] <= event_time + pd.Timedelta(seconds=(time_period / 2)))]
    df_temporary["ESM_timestamp"] = row["ESM_timestamp"]

    #concatenate df_results with df_temporary
    df_results = pd.concat([df_results, df_temporary])

# merge participant IDs
#df_results = merge_participantIDs(df_results, users, device_id_col = "2", include_cities = False)

print("Unique ESM_timestamp values after cutting to 90 seconds around events for locations: " + str(df_results["ESM_timestamp"].nunique()))

#testarea 2
#convert count number of values in column 1 for each ESM_timestamp in df
df_test = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/FINAL_locations_esm_timeperiod_5 min.csv")
test = df_test2.groupby("ESM_timestamp")["timestamp"].count()

# only keep rows with column 1 == "test"
df_test2 = df.copy()
#convert timestamp into datetime
df_test2["timestamp"] = pd.to_datetime(df_test2["1"], unit = "ms")
df_test2["ESM_timestamp"] = pd.to_datetime(df_test2["ESM_timestamp"], unit = "ms")

df_test2 = df_test2[df_test2["ESM_timestamp"] == pd.to_datetime("2022-07-21 18:03:12.856999936")]
# count number of unique column 2 values
df_test2["2"].nunique()
# delete  columns "Unnamed: 0" and "0"
df_test2 = df_test2.drop(labels=["Unnamed: 0", "0"], axis=1)
#drop duplicates
df_test3 = df_test2.drop_duplicates()
df_test2["ESM_timestamp"].nunique()

#temporary 3

dir_databases = "/Volumes/INTENSO/In Usage new/Databases/db_iteration1_20220723"
file_list = [i for i in os.listdir(dir_databases) if not i.startswith(".")]  # for Mac

for file in file_list:
    df_temporary = pd.read_csv(dir_databases + "/" + file)
    #rename columns so they match the ones from the MySQL database
    df_temporary.rename(columns={df_temporary.columns[1]: '0'}, inplace=True)
    df_temporary.rename(columns={df_temporary.columns[2]: '1'}, inplace=True)
    df_temporary.rename(columns={df_temporary.columns[3]: '2'}, inplace=True)
    df_temporary.rename(columns={df_temporary.columns[4]: '3'}, inplace=True)
    df_temporary.to_csv(dir_databases + "/" + file, index=False)
    print("Finished " + str(file))

#temporary 2
esm_entry = esm_all_transformed.iloc[0]
esm_entry = esm_all_transformed.iloc[1107]

sensor_data_in_time_period = sensor_all[
    (sensor_all["2"] == esm_entry["device_id"]) & (sensor_all["1"] >= esm_entry["timestamp"] - time_period_unix) & (
                sensor_all["1"] <= esm_entry["timestamp"] + time_period_unix)].copy()
sensor_data_in_time_period["ESM_timestamp"] = esm_entry["timestamp"]

sensor_data_in_time_period["DateTime"] = pd.to_datetime(sensor_data_in_time_period["1"], unit='ms')
sensor_data_in_time_period["ESM_timestamp"] = pd.to_datetime(sensor_data_in_time_period["ESM_timestamp"], unit='ms')
len(sensor_data_in_time_period)



#temporary
paths_intermediate2 = []
for root, dirs, files in os.walk(dir_databases):  # iterate through different subfolders
    for subfolder in dirs:
        time_start = time.time()
        path_sensor = dir_databases + "/" + subfolder + "/" + sensor + ".csv"

        if os.path.exists(path_sensor):
            paths_intermediate2.append(dir_databases + "/TEMPORARY_" + sensor + "_" + subfolder + "_esm_timeperiod_" + str(time_period) + " min.csv")
        else:
            print("continued")


df_final = pd.DataFrame()
dir_databases = "/Users/benediktjordan/Documents/MTS/Iteration01/Data"
time_period = 5
sensor = "linear_accelerometer"
paths_intermediate = ["/Users/benediktjordan/Documents/MTS/Iteration01/Data/TEMPORARY_linear_accelerometer_db_iteration1_20220702_esm_timeperiod_5 min.csv",
                                          "/Users/benediktjordan/Documents/MTS/Iteration01/Data/TEMPORARY_linear_accelerometer_db_iteration1_20220704_esm_timeperiod_5 min.csv",
                                          "/Users/benediktjordan/Documents/MTS/Iteration01/Data/TEMPORARY_linear_accelerometer_db_iteration1_20220707_esm_timeperiod_5 min.csv",
                                          "/Users/benediktjordan/Documents/MTS/Iteration01/Data/TEMPORARY_linear_accelerometer_db_iteration1_20220707_2_esm_timeperiod_5 min.csv",
                                          "/Users/benediktjordan/Documents/MTS/Iteration01/Data/TEMPORARY_linear_accelerometer_db_iteration1_20220708_esm_timeperiod_5 min.csv",
                                          "/Users/benediktjordan/Documents/MTS/Iteration01/Data/TEMPORARY_linear_accelerometer_db_iteration1_20220709_esm_timeperiod_5 min.csv",
                                          "/Users/benediktjordan/Documents/MTS/Iteration01/Data/TEMPORARY_linear_accelerometer_db_iteration1_20220710_esm_timeperiod_5 min.csv",
                                          "/Users/benediktjordan/Documents/MTS/Iteration01/Data/TEMPORARY_linear_accelerometer_db_iteration1_20220712_esm_timeperiod_5 min.csv"]
sensor = "gyroscope"
paths_intermediate = ["/Users/benediktjordan/Documents/MTS/Iteration01/Data/TEMPORARY_gyroscope_db_iteration1_20220702_esm_timeperiod_5 min.csv",
                                          "/Users/benediktjordan/Documents/MTS/Iteration01/Data/TEMPORARY_gyroscope_db_iteration1_20220704_esm_timeperiod_5 min.csv",
                                          "/Users/benediktjordan/Documents/MTS/Iteration01/Data/TEMPORARY_gyroscope_db_iteration1_20220707_esm_timeperiod_5 min.csv",
                                          "/Users/benediktjordan/Documents/MTS/Iteration01/Data/TEMPORARY_gyroscope_db_iteration1_20220707_2_esm_timeperiod_5 min.csv",
                                          "/Users/benediktjordan/Documents/MTS/Iteration01/Data/TEMPORARY_gyroscope_db_iteration1_20220708_esm_timeperiod_5 min.csv",
                                          "/Users/benediktjordan/Documents/MTS/Iteration01/Data/TEMPORARY_gyroscope_db_iteration1_20220709_esm_timeperiod_5 min.csv",
                                          "/Users/benediktjordan/Documents/MTS/Iteration01/Data/TEMPORARY_gyroscope_db_iteration1_20220710_esm_timeperiod_5 min.csv"]


for path_intermediate in paths_intermediate:
    # check if path_intermediate contains any data; if not -> continue
    try:
        df_intermediate = pd.read_csv(path_intermediate)
        df_final = pd.concat([df_final, df_intermediate], ignore_index=True, sort=False)
        print("finished "+ path_intermediate)
    except:
        print("Error for file: " + path_intermediate)
        continue
df_final.to_csv(dir_databases + "/" + sensor + "_esm_timeperiod_" + str(time_period) + " min.csv", index=False)

#endregion

#region transform unix timestamp into datetime
## the function can be found in "general_functions"

dir_sensorfiles = "/Users/benediktjordan/Documents/MTS/Iteration01/Data/"
file_list = [i for i in os.listdir(dir_sensorfiles) if not i.startswith(".")]  # for Mac

# define names of columns which should be converted from unix to datetime
column_names = ["1","ESM_timestamp", "ESM_location_time", "ESM_bodyposition_time", "ESM_activity_time",
                "ESM_smartphonelocation_time", "ESM_aligned_time", "timestamp"]

# iterate through sensor files and a) convert JSON columns into multiple columns and
# b) convert unix timestamp into datetime
for file in file_list:
    if file.endswith(".csv"):
        time_start = time.time()
        path_df = dir_sensorfiles + file
        file = file[6:] # remove "FINAL_" from file name

        # convert JSON column into multiple columns
        df = convert_json_to_columns(path_df, sensor = file[0:3] ,json_column_name="3")

        # rename sensor specific timestamp column into "timestamp"
        column_name = file[0:3] + "_timestamp"
        df = df.rename(columns={column_name: "timestamp"})

        # convert unix timestamp into datetime
        for column_name in column_names: # column_names are defined in the beginning
            df = convert_unix_to_datetime(df, column_name)

        df.to_csv(dir_sensorfiles + file + "_JSONconverted.csv", index=False)
        time_end = time.time()
        print("finished " + file + " in " + str((time_end - time_start) / 60) + " minutes")


#endregion

