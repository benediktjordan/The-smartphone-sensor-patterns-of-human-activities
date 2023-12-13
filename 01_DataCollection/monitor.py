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

#region monitor (during data collection) to ensure correct data collection functionality

def overview_sensor_perdatabase(base_path, db_name, users, sensors_and_frequencies):

    path = str(base_path + db_name + "/" ) #path of tables/sensors
    list_sensor = []
    list_frequency = []
    list_user_name = []
    list_duplicate_number = []
    list_total_number = []
    list_time_beginning_formatted = []
    list_time_end_formatted = []
    list_expected_number = []
    list_total_minutes = []
    list_completeness_percentage = []

    for index_sensors, sensor_frequency in sensors_and_frequencies.iterrows():
        start_time = time.time()

        sensor = sensor_frequency[0]
        frequency = sensor_frequency[1]

        #load table
        try:
            df_table = pd.read_csv (str(path+sensor+".csv")) # sometimes a sensor is not captured
            test = df_table["2"] #sometimes a sensor file is empty (i.e. studentlife_audio)
        except:
            list_sensor.append(sensor)
            list_frequency.append("no data found")
            list_user_name.append(-1)
            list_duplicate_number.append(-1)
            list_total_number.append(-1)
            list_time_beginning_formatted.append(-1)
            list_time_end_formatted.append(-1)
            list_expected_number.append(-1)
            list_total_minutes.append(-1)
            list_completeness_percentage.append(-1)
            continue
        else:
            df_table = pd.read_csv(str(path + sensor + ".csv"))

            for index_users, user in users.iterrows():
                user_id = user[1]
                user_name = user[0]
                data_user = df_table.loc[df_table['2'] == str(user_id)]  # filter sensor data from only this user
                if len(data_user) == 0: #in case there is no data for this user
                    continue
                else:
                    list_sensor.append(sensor)
                    list_frequency.append(frequency)
                    list_user_name.append(user_name)


                    # delete duplicates and compute number of duplicates
                    data_without_duplicates_user  = data_user.drop_duplicates()
                    list_duplicate_number.append(len(data_user)-len(data_without_duplicates_user))

                    #compute different numbers regarding sensor & add them to list
                    list_total_number.append(len(data_without_duplicates_user))
                    time_beginning_user = data_without_duplicates_user.iloc[0][2]
                    list_time_beginning_formatted.append(datetime.datetime.fromtimestamp(time_beginning_user/1000).strftime('%Y-%m-%d %H:%M:%S.%f'))
                    time_end_user = data_without_duplicates_user.iloc[-1][2]
                    list_time_end_formatted.append(datetime.datetime.fromtimestamp(time_end_user/1000).strftime('%Y-%m-%d %H:%M:%S.%f'))
                    list_total_minutes.append((time_end_user-time_beginning_user)/60000)
                    list_expected_number.append(list_total_minutes[-1]*60*frequency) #since 60 seconds per minute
                    list_completeness_percentage.append(list_total_number[-1]/list_expected_number[-1])
        del df_table
        end_time = time.time()
        print("For sensor "+ str(sensor) + " it took "+ str((end_time-start_time) / 60) + " minutes")

    df_results = pd.DataFrame(list(zip(list_sensor, list_frequency,list_user_name, list_time_beginning_formatted,
                                       list_time_end_formatted,list_total_minutes, list_duplicate_number, list_total_number, list_expected_number,
                                       list_completeness_percentage)),
                              columns = ["Sensor", "Frequency", "Name", "Recording Beginning",
                                         "Recording End", "Total Time (minutes", "Number of Duplicates", "Number of Entries",
                                         "Expected Number of Entries","Completenes Percentage"])

    df_results.to_csv(base_path + db_name + "/" + str(db_name) +"_DataSanityAnalysis.csv")

# region iterate through sensors and users
base_path = "/Volumes/TOSHIBA EXT/InUsage/Databases/"
for root, dirs, files in os.walk(base_path):  # iterate through different subfolders
    for subfolder in dirs:
        overview_sensor_perdatabase(base_path, subfolder, users, sensors_and_frequencies)

## alternative: use list of folders
list_folders = ["db_iteration1_20220628", "db_iteration1_20220707", "db_iteration1_20220709",
                "db_iteration1_20220708","db_iteration1_20220701", "nlbbtest", "db_iteration1_20220707_2",
                "db_iteration1_20220712", "db_iteration1_20220717", "old_data"]
for folder in list_folders:
    overview_sensor_perdatabase(base_path, folder, users, sensors_and_frequencies)
    print("Success for " + str(folder))
# endregion

db_name = "db_iteration1_20220628"
path = str(base_path + db_name + "/" )
sensor = "gravity_from_S3"
df_table = pd.read_csv(str(path + sensor + ".csv"))
# endregion

#region create basic df with all users & sensors
def overview_sensors_alldatabases_initialize(users, sensors_and_frequencies, df_first_overview):

    name = []
    sensor = []
    frequency = []
    recording_beginning = []
    recording_end = []
    total_time = []
    duplicates = []
    entries = []
    entries_exptected = []
    completeness = []
    for user in users["Name"]:
        for index, sensor_row in sensors_and_frequencies.iterrows():
            name.append(user)
            sensor.append(sensor_row[0])
            frequency.append(sensor_row[1])

            subset = df_first_overview[(df_first_overview['Name'] == user) & (df_first_overview['Sensor'] == sensor_row[0])]
            if (len(subset.index) == 1):
                row = df_first_overview.iloc[[subset.index[0]]]
                recording_beginning.append(row["Recording Beginning"].iloc[0])
                recording_end.append(row["Recording End"].iloc[0])
                total_time.append(row["Total Time (minutes"].iloc[0])
                duplicates.append(row["Number of Duplicates"].iloc[0])
                entries.append(row["Number of Entries"].iloc[0])
                entries_exptected.append(row["Expected Number of Entries"].iloc[0])
                completeness.append(row["Completenes Percentage"].iloc[0])
            else:
                recording_beginning.append(0)
                recording_end.append(0)
                total_time.append(0)
                duplicates.append(0)
                entries.append(0)
                entries_exptected.append(0)
                completeness.append(0)

    df_all = pd.DataFrame(list(zip(name,sensor,frequency,recording_beginning,recording_end,total_time,duplicates,
                                   entries, entries_exptected, completeness)),
                          columns = ["User", "Sensor", "Frequency", "Recording Beginning", "Recording End", "Total Time",
                                     "Duplicates", "Entries", "Entries Expected", "Completeness"])
    return df_all

#df_all = join_databases_dataoverviews_initialize(users, sensors_and_frequencies, df_test)
#endregion

#region create function to add information to this basic data frame

def overview_sensors_alldatabases(df_general, df_to_add):
    entries_changed = 1
    indices_used = []
    for index, row in df_general.iterrows():
        subset = df_to_add[(df_to_add['Name'] == row["User"]) & (df_to_add['Sensor'] == row["Sensor"])] #check if name and sensor is in df
        if (len(subset.index) == 1):
            df_general["Recording End"][index] = subset["Recording End"].iloc[0]
            # check if recording beginning time is there or else add it
            if (df_general["Recording Beginning"][index]==0):
                df_general["Recording Beginning"][index] = subset["Recording Beginning"].iloc[0]
            total_time = divmod((datetime.datetime.strptime(df_general["Recording End"][index], '%Y-%m-%d %H:%M:%S.%f') - datetime.datetime.strptime(df_general["Recording Beginning"][index], '%Y-%m-%d %H:%M:%S.%f')).total_seconds(),60)[0]
            df_general["Total Time"][index] = total_time
            df_general["Duplicates"][index] = (df_general["Duplicates"][index]+subset["Number of Duplicates"].iloc[0])
            df_general["Entries"][index] = (df_general["Entries"][index]+subset["Number of Entries"].iloc[0])
            df_general["Entries Expected"][index] = df_general["Frequency"][index]*total_time*60 #minutes * frequency in Hz * 60 seconds
            df_general["Completeness"][index] = df_general["Entries"][index]/df_general["Entries Expected"][index]
            indices_used.append(subset.index[0])
            entries_changed = entries_changed+1
            #print("Iterated through index " + str(index))

    if (entries_changed == len(df_to_add)):
        print("All Entries have been successfully changed")
    else:
        print("There was an error somewhere")
        print("Entries Changed: " + str(entries_changed))
        print("Length of DataFrame: " + str(len(df_to_add)))

    return df_general

#endregion

#region iterate through databases and create data sanity overview
#df_all = pd.read_csv (str(base_path+"DataSanityAnalysis_allDatabases.csv"))
#db_name = "db_iteration1_20220707_2"
#path = base_path+db_name+"/"+db_name+"_DataSanityAnalysis.csv"
#df_to_add = pd.read_csv(path)
#df_all = join_databases_dataoverviews(df_all, df_to_add)

# iterate through all databases and functions
database_list =  ["nlbbtest",
                  "nlbbtest2",
                  "nlbbtest4",
                  "nlbbtest5",
                  "db_iteration1_20220624",
                  "db_iteration1_20220628",
                  "db_iteration1_20220701",
                  "db_iteration1_20220702",
                  "db_iteration1_20220704",
                  "db_iteration1_20220707",
                  "db_iteration1_20220707_2",
                  "db_iteration1_20220708",
                  "db_iteration1_20220709",
                  "db_iteration1_20220710",
                  "db_iteration1_20220712",
                  "db_iteration1_20220717"
                  ]


base_path = "E:/InUsage/Databases/"

counter = 0
for db_name in database_list:
    path = base_path+db_name+"/"+db_name+"_DataSanityAnalysis.csv"
    if (counter == 0):
        df_first_overview = pd.read_csv(path)
        df_all = overview_sensors_alldatabases_initialize(users, sensors_and_frequencies, df_first_overview)
        print("The overview has been initialized with the first database overview")
    else:
        df_to_add = pd.read_csv(path)
        df_all = overview_sensors_alldatabases(df_all, df_to_add)
        print(str(db_name) + " has been successfully added")
    counter = counter + 1

df_all.to_csv(base_path +"DataSanityAnalysis_allDatabases.csv")
# endregion

