# region import libraries
import numpy as np
import pandas as pd
import numpy as np
import pickle
import tsfresh
#endregion

# load ESM file
path_esm ="/Users/benediktjordan/Documents/MTS/Iteration01/Data/esm_all_transformed.csv"
df_esm = pd.read_csv(path_esm, parse_dates=["timestamp"])

df_esm["activity"].value_counts()
df_esm["bodyposition"].value_counts().sum()



#region transform labels for my activity classification
# create dictionary which maps activities to different activity levels

class label_transformation:
    def create_activity_dataframe(df_esm, human_motion, humanmotion_general, humanmotion_specific, before_after_sleep,
                                  on_the_toilet_sittingsomewhereelse, on_the_toilet, publictransport_non_motorized, publictransport,
                                  location, smartphonelocation, aligned):

        # special case for "human_motion", since more than one column in df_esm have to be evaluated to create each label
        #iterate through dictionary human_motion
        df_esm["label_human motion"] = np.nan
        for key in human_motion:
            df_map = human_motion[key]
            # iterate through dataframe
            for index, row in df_map.iterrows():
                if len(row) == 1:
                    #insert in df_esm["human motion") the key in case in the df_esm column == df_map.columns[0] the values == row.iloc[0] and in the df_esm column == df_map.columns[1] the values == row.iloc[1]
                    df_esm.loc[(df_esm[df_map.columns[0]] == row.iloc[0]), "label_human motion"] = key

                if len(row) == 2:
                    #insert in df_esm["human motion") the key in case in the df_esm column == df_map.columns[0] the values == row.iloc[0] and in the df_esm column == df_map.columns[1] the values == row.iloc[1]
                    df_esm.loc[(df_esm[df_map.columns[0]] == row.iloc[0]) & (df_esm[df_map.columns[1]] == row.iloc[1]), "label_human motion"] = key

                if len(row) == 3:
                    #insert in df_esm["human motion") the key in case in the df_esm column == df_map.columns[0] the values == row.iloc[0] and in the df_esm column == df_map.columns[1] the values == row.iloc[1]
                    df_esm.loc[(df_esm[df_map.columns[0]] == row.iloc[0]) & (df_esm[df_map.columns[1]] == row.iloc[1]) & (df_esm[df_map.columns[2]] == row.iloc[2]), "label_human motion"] = key


        # create dataframe with details of new columns
        new_columns = [["label_human motion - general", "humanmotion_general", "bodyposition"],
                       ["label_human motion - specific", "humanmotion_specific", "bodyposition"],
                       ["label_before and after sleep", "before_after_sleep", "activity"],
                       ["label_on the toilet", "on_the_toilet_sittingsomewhereelse", "bodyposition"],
                       ["label_on the toilet", "on_the_toilet", "activity"],
                       ["label_public transport", "publictransport_non_motorized", "bodyposition"],
                       ["label_public transport", "publictransport", "location"],
                       ["label_location", "location", "location"],
                       ["label_smartphone location", "smartphonelocation", "smartphonelocation"],
                       ["label_aligned", "aligned", "aligned"]]

        df_new_columns = pd.DataFrame(new_columns, columns = ["column name", "column dictionary", "column to map"])

        # iterate through dataframe and create new columns
        for index, row in df_new_columns.iterrows():
            # check if column_name already exists; if it does, then just add the data frome the current column dictionary (not replace it)
            # this is relevant i.e. for "on the toilet" as it is a combination of two columns
            if row["column name"] not in df_esm.columns:
                # create new column
                df_esm[row["column name"]] = df_esm.apply(lambda x : x[row["column to map"]] if (x[row["column to map"]] in  eval(row["column dictionary"])) else np.nan, axis = 1)
                # replace values in new column by the ones which are defined in the mapping dictionary
                df_esm[row["column name"]] = df_esm[row["column name"]].replace(eval(row["column dictionary"]))

            else:
                # add data from current column dictionary to existing column
                df_esm[row["column name"]] = df_esm.apply(lambda x : x[row["column to map"]] if (x[row["column to map"]] in  eval(row["column dictionary"])) else x[row["column name"]], axis = 1)
                # replace values in new column by the ones which are defined in the mapping dictionary
                df_esm[row["column name"]] = df_esm[row["column name"]].replace(eval(row["column dictionary"]))
                print("column already exists: " + row["column name"])

        return df_esm

    def create_activity_dictionary_from_dataframe(df_esm_including_activity_classes_dataframe):
        df_esm = df_esm_including_activity_classes_dataframe

        # create timestamp_datetime column
        df_esm["timestamp_datetime"] = pd.to_datetime(df_esm["timestamp"], unit = "ms")

        # convert df_esm into dict (set timestamp as keys) and save it
        df_esm = df_esm.set_index("timestamp_datetime")
        df_esm_dict = df_esm.to_dict(orient = "index")

        return df_esm_dict




# check if there are duplicates in timestamp
test = df_esm["timestamp"].value_counts()
test.describe()
