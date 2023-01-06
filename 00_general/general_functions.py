#region transform unix timestamp into datetime

def convert_unix_to_datetime(df, column_name):
    df[column_name] = pd.to_datetime(df[column_name], unit='ms')
    return df

#endregion

# region data preparation: label sensorfiles
#function to label sensorfiles
#updated version of labeling_sensor_df
def labeling_sensor_df(df_sensor, dict_label, label_column_name, ESM_identifier_column = "ESM_timestamp"):
    # create dataframe with onl ESM_identifier_column and label_column_name
    df_label = pd.DataFrame()
    df_label[ESM_identifier_column] = df_sensor[ESM_identifier_column].copy()
    # convert df_label to DataFrame
    # iterate through df_sensor and find labels for each row
    df_label[label_column_name] = np.nan
    # iterate through df_label and find labels for each row in dict_label and add them to df_label. Use apply function
    df_label[label_column_name] = df_label.apply(lambda row: dict_label[pd.Timestamp(row[ESM_identifier_column])][label_column_name], axis=1)

    #for index, row in df_label.iterrows():
    #    # find label for each row
    #    label = dict_label[pd.Timestamp(row[ESM_identifier_column])][label_column_name]
    #    # add label to df_sensor
    #    df_label.at[index, label_column_name] = label
        #print("label added for row ", index)
    # delete ESM_identifier_column
    df_label.drop(columns=[ESM_identifier_column], inplace=True)
    # add label column in df_label to df_sensor based on index
    df_sensor = pd.concat([df_sensor, df_label], axis=1)

    return df_sensor

#endregion