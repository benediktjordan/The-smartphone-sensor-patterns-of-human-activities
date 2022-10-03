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


# This is a change comment

# user and sensor database
users = pd.DataFrame([["Simone_1","61bf23e5-0a6b-4d3c-b393-1a23d4f64e88"],
                      ["Simone_2","4c4e5063-1b23-4dfc-886d-c6a202225ed6"],
                      ["Simone_3","53c73807-3756-4303-b4dc-5e7e232e528c"],
                      ["Simone_4", "f83ed117-9279-4ef8-ab74-83d7b8b268b8"],
                      ["Tina", "0d620b8a-c2d4-48fc-9c75-80ce80aeea3e"],
                      ["Tina_2", "3936a8f9-8be0-4523-bd0f-2a03943cb5f0"],
                      ["Lea","590f0faf-d932-4a57-998d-e3da667a91dc"],
                      ["Lotte","410df445-af3d-4cf7-8bf1-f92160f1c41f"],
                      ["Bella","ad064def-9d72-44ff-96b2-f3b3d90d1079"],
                      ["Madleine","36cf6eb2-9d88-46db-a90c-3f24cb4b7228"],
                      ["Miranda","cf2dfa9b-596b-4a72-a4da-cb3777a31cc7"],
                      ["Lena", "212a5ebe-0714-47ac-a887-964c24e0ae43"],
                      ["Paul", "15cdbd7c-132e-4bb8-9dab-192cf909daec"],
                      ["Selcuk", "2294d0b0-67ef-4af2-8ffb-69db607920c9"],
                      ["Selcuk_2", "c838b909-f782-4441-aa3c-10c6c7765ba3"],
                      ["Rosa", "84afe4cb-3572-46bc-bc29-d982ac375341"],
                      ["Rosa_2", "e6c1d093-148e-47f8-8054-6663dc5c366a"],
                      ["Bini", "6388b5d9-367b-427e-a2f5-912014c69a5e"],
                      ["Tanzi", "e9d3ed5e-1d52-445c-82ac-8bbe8066b3d7"],
                      ["Pauli", "6ab9716e-e6d8-4492-ad86-f051a9a4b62a"],
                      ["Margherita ?", "b23b3f4e-7fc1-452f-be16-b9388451f3f6"],
                      ["Margherita_2", "25f1657f-5a39-4dba-8a3b-e6efbfec0e4d"],
                      ["Unknown", "3936a8f9-8be0-4523-bd0f-2a03943cb5f0"],
                      ["Felix", "b7b013b7-f78c-4325-a7ab-2dfc128fba27"],
                      ["Benedikt_1", "8206b501-20e7-4851-8b79-601af9e4485f"],
                      ["Benedikt_2", "f1dc0e69-a548-4771-85f9-28db9060d4c6"],
                      ["Benedikt_3", "eab5ef78-8eb2-4587-801f-834fc1f86f31"],
                      ["Benedikt_4", "f1db4f27-2e1c-4558-85a1-b43ef2c5af59"],
                      ["Benedikt_5 (vorher Unknown_2)", "57fc9641-9f4d-409b-bffd-f333b01c33c9"],
                      ["Benedikt_secondiPhone", "4c1db32b-48fc-4fa6-a4fe-c44f079b7ca4"],
                      ["Benedikt_tablet","0960f02f-8c67-486c-b8db-7850d4a7070b"]], columns = ["Name", "ID"])

sensors_and_frequencies = pd.DataFrame([["barometer",0.1],
                                        ["accelerometer",10],
                                          ["battery",0], #event based
                                          ["battery_charges",0], #event based
                                          ["battery_discharges",0], #event based
                                          ["bluetooth", 0.0333333],
                                          ["esm", 0.00055555555], #12 times a day while every time I get 4 events
                                          ["gravity", 10],
                                          ["gyroscope", 10],
                                          ["ios_status_monitor", 0.01666666666],
                                          ["linear_accelerometer", 10],
                                          ["locations", 1],
                                          ["magnetometer", 10],
                                          ["network", 0], #event based
                                          ["plugin_ambient_noise", 0.01666666666], #once per minute
                                          ["plugin_device_usage", 0], #event based
                                          ["plugin_ios_activity_recognition", 0.1],
                                          ["plugin_ios_esm", 0.00055555555], #12 times a day while every time I get 4 events
                                          ["plugin_openweather", 0.00055555555], #every 30 minutes
                                          ["plugin_studentlife_audio", 0.00055555555], #every 4 minutes
                                          ["push_notification", 0.00013888888], #18 times a day
                                          ["screen", 0], #event based
                                          ["rotation", 10],
                                          ["sensor_wifi", 1],
                                          ["significant_motion", 0],
                                          ["timezone", 0.00027777777] #once per hour
                                          ], columns = ["Sensor", "Frequency (in Hz)"])




#region Monitor (during data collection)

# region create "Data Sanity" sheet to get an overview over data completeness and duplicates
# region create data overview csv for every database
#db_name = "db_iteration1_20220712"




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


#endregion

#region Download
db_name = "db_iteration1_20220723"
path_save_database = "D:/MasterThesis @nlbb/Iteration01/I1_Data Collection/AWARE/Databases/"

#region create connection to database

db = mysql.connector.connect(
    host = "nlbbtest.cmucwdhldu8x.us-east-2.rds.amazonaws.com",
    user = "user1",
    password = "nlbbtestpassword123",
    port = "3306",
    database = db_name
)

mycursor = db.cursor(buffered=True)

#endregion

#region import and save tables
# get list of table names
statement = str("SELECT table_name, table_rows FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = '" + db_name + "'")
mycursor.execute(statement)
tables_included = mycursor.fetchall()

tables_included_final = []
for i in tables_included:
    add = i[0].decode()
    tables_included_final.append(add)
#df_tables_included = pd.DataFrame(tables_included)

#create folder to which to save the data
os.mkdir(path_save_database+db_name)
#df_tables_included = df_tables_included.drop([0,1,2,3,4,5,6,7,8,9,10,11])

# load every table into dataframe and save it as a csv
counter = 0

for table in tables_included_final:
    start_time = time.time()

    # start database connection
    db = mysql.connector.connect(
        host="nlbbtest.cmucwdhldu8x.us-east-2.rds.amazonaws.com",
        user="user1",
        password="nlbbtestpassword123",
        port="3306",
        database=db_name
    )
    mycursor = db.cursor(buffered=True)

    #download tables
    statement = str("SELECT * FROM "+ table)
    try:
        mycursor.execute(statement)
        table_data = mycursor.fetchall()
    except:
        print("Couldn´t get any data for " + str(table))
        continue
    else:
        mycursor.execute(statement)
        table_data = mycursor.fetchall()
        df_table_data = pd.DataFrame(table_data)
        df_table_data.to_csv(path_save_database + db_name + "/" + str(table) + ".csv")

    #count total rows of this table
    print("Table " + str(table) + " has been successfullly saved.")
    print("Number of SQL rows: " + str(df_tables_included[1][counter]))
    print("Number of exported rows: " + str(len(df_table_data)))

    #start database connection
    db = mysql.connector.connect(
        host="nlbbtest.cmucwdhldu8x.us-east-2.rds.amazonaws.com",
        user="user1",
        password="nlbbtestpassword123",
        port="3306",
        database=db_name
    )
    mycursor = db.cursor(buffered=True)

    #try to get total count of sensor data
    statement = str("select COUNT(*) from " + table)

    try:
        mycursor.execute(statement)
        counted_rows = mycursor.fetchall()
    except:
        print("Couldn´t get number of rows for " + str(table))
        continue
    else:
        mycursor.execute(statement)
        counted_rows = mycursor.fetchall()
        print("Number of counted rows on SQL " + str(counted_rows[0][0]))

    end_time = time.time()
    print("--- %s minutes ---" % ((time.time() - start_time) / 60))
    counter = counter + 1





#endregion

# region Backup: load number of rows for every table and save into dataframe
table_list = []
count_row_list = []
for table in df_tables_included[0]:
    start_time = time.time()
    #start database connection
    db = mysql.connector.connect(
        host="nlbbtest.cmucwdhldu8x.us-east-2.rds.amazonaws.com",
        user="user1",
        password="nlbbtestpassword123",
        port="3306",
        database=db_name
    )

    mycursor = db.cursor(buffered=True)
    statement = str("select COUNT(*) from " + table)
    mycursor.execute(statement)
    counted_rows = mycursor.fetchall()
    table_list.append(table)
    count_row_list.append(counted_rows)
    print("Finished: " + str(table))
df_countedrows = pd.DataFrame(list(zip(table_list, count_row_list)),
                              columns = ["Sensors", "Total Number"])

#rint("Number of counted rows on SQL " + str(counted_rows[0][0]))
#endregion

#endregion

#region Transform


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

#region ESM: join all ESM files and transform them so that I have one row per ESM event

# region main functions
def join_sensor_files(dir_databases, sensor):
    counter = 0
    for root, dirs, files in os.walk(dir_databases):  # iterate through different subfolders
        for subfolder in dirs:
            path_sensor = dir_databases + "/" + subfolder + "/" + sensor + ".csv"
            if os.path.exists(path_sensor):
                if counter == 0:
                    sensor_all = pd.read_csv(path_sensor)
                    counter += 1
                    print("This file was used as the base: " + str(path_sensor))
                else:
                    sensor_all = sensor_all.append(pd.read_csv(path_sensor))
                    print("len of sensor_all is : " + str(len(sensor_all)))
                    print("This files was added: " + str(path_sensor))
                    counter += 1
            else:
                continue

    return sensor_all

# endregion

# region manipulate ESM
# region labels: join all ESM files

dir_databases = "/Volumes/INTENSO/In Usage new/Databases"
sensor = "esm"
esm_all = join_sensor_files(dir_databases, sensor)

esm_all.to_csv("/Volumes/INTENSO/In Usage new/Databases/esm_all.csv")

# endregion

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

#endregion

#region filter/select only data in x minute range around ES events

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

#region convert JSON column into multiple columns & add sensor identifier

def convert_json_to_columns(path_df, sensor, json_column_name):
    df = pd.read_csv(path_df, dtype={"0": 'int64', "1": 'int64', "2": object, "3": object, "ESM_timestamp": 'float64',
                                                    "ESM_location": object, "ESM_location_time": "float64",
                                                    "ESM_bodyposition": object, "ESM_bodyposition_time": "float64",
                                                    "ESM_activity": object, "ESM_activity_time": "float64",
                                                    "ESM_smartphonelocation": object, "ESM_smartphonelocation_time": "float64",
                                                    "ESM_aligned": object, "ESM_aligned_time": "float64",})
    df[json_column_name] = df[json_column_name].apply(lambda x: json.loads(x))
    # add sensor identifier to every JSON column
    df = pd.concat([df, df[json_column_name].apply(pd.Series).add_prefix(sensor + "_") ], axis=1)

    return df

# NOTE: function is applied in next section
# double check
file = file_list[0]
df_old = pd.read_csv(dir_sensorfiles + file)
df = pd.read_csv(dir_sensorfiles + file + "_JSONconverted.csv")

#endregion

#region transform unix timestamp into datetime
def convert_unix_to_datetime(df, column_name):
    df[column_name] = pd.to_datetime(df[column_name], unit='ms')
    return df

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

#endregion

#endregion

#endregions

#endregion


