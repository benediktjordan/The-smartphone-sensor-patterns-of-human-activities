# region import libraries
import numpy as np
import pandas as pd
import numpy as np
import pickle
import tsfresh
#endregion

# in this class, the context classes are added based on the ESM answers
class label_transformation:

    # add context classes to df_esm (the dataframe which contains the esm data): used in next function
    ## Note: this works for laboratory data; and also for naturalistic -> human motion data; for the other
    ## naturalistic contexts, it doesnt work (use the next function)
    def add_activity_classes(df_esm, dict_mapping, label_column_name):

        df_esm[label_column_name] = np.nan
        for key in dict_mapping:
            df_map = dict_mapping[key]
            # iterate through dataframe
            for index, row in df_map.iterrows():
                if len(row) == 1:
                    # insert in df_esm["human motion") the key in case in the df_esm column == df_map.columns[0] the values == row.iloc[0] and in the df_esm column == df_map.columns[1] the values == row.iloc[1]
                    df_esm.loc[(df_esm[df_map.columns[0]] == row.iloc[0]), label_column_name] = key

                if len(row) == 2:
                    # insert in df_esm["human motion") the key in case in the df_esm column == df_map.columns[0] the values == row.iloc[0] and in the df_esm column == df_map.columns[1] the values == row.iloc[1]
                    df_esm.loc[(df_esm[df_map.columns[0]] == row.iloc[0]) & (
                                df_esm[df_map.columns[1]] == row.iloc[1]), label_column_name] = key

                if len(row) == 3:
                    # insert in df_esm[label_column_name) the key in case in the df_esm column == df_map.columns[0] the values == row.iloc[0] and in the df_esm column == df_map.columns[1] the values == row.iloc[1]
                    df_esm.loc[
                        (df_esm[df_map.columns[0]] == row.iloc[0]) & (df_esm[df_map.columns[1]] == row.iloc[1]) & (
                                    df_esm[df_map.columns[2]] == row.iloc[2]), label_column_name] = key
        return df_esm


    # add context classes to df_esm (the dataframe which contains the esm data): main function
    def create_activity_dataframe(df_esm, human_motion, humanmotion_general, humanmotion_specific, before_after_sleep, before_after_sleep_updated,
                                  on_the_toilet_sittingsomewhereelse, on_the_toilet, publictransport_non_motorized, publictransport,
                                  location, smartphonelocation, aligned):

        # special case for "human_motion" and "before_after_sleep_updated", since more than one column in df_esm have to be evaluated to create each label
        #iterate through dictionary human_motion and dictionary before_after_sleep_updated
        df_esm = label_transformation.add_activity_classes(df_esm, human_motion, "label_human motion")
        df_esm = label_transformation.add_activity_classes(df_esm, before_after_sleep_updated, "label_sleep")

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

    # create dictionary from dataframe
    def create_activity_dictionary_from_dataframe(df_esm_including_activity_classes_dataframe):
        df_esm = df_esm_including_activity_classes_dataframe

        # create timestamp_datetime column
        # check if column "timestamp" exists
        if "timestamp" in df_esm.columns: #this case is for iteration 01 data
            df_esm["timestamp_datetime"] = pd.to_datetime(df_esm["timestamp"], unit = "ms")
        elif "ESM_timestamp" in df_esm.columns: #this case is for iteration 02 data
            df_esm["timestamp_datetime"] = pd.to_datetime(df_esm["ESM_timestamp"], unit = "ms")

        # convert df_esm into dict (set timestamp as keys) and save it
        df_esm = df_esm.set_index("timestamp_datetime")
        df_esm_dict = df_esm.to_dict(orient = "index")

        return df_esm_dict



