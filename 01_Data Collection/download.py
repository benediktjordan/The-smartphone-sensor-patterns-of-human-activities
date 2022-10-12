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
