# import libraries
import numpy as np
import pandas as pd
import numpy as np

# load ESM file
path_esm ="/Users/benediktjordan/Documents/MTS/Iteration01/Data/esm_all_transformed.csv"
df_esm = pd.read_csv(path_esm, parse_dates=["timestamp"])

df_esm["activity"].value_counts()
df_esm["bodyposition"].value_counts().sum()



# transform labels for my activity classification
# region: human motion
# create dictionary which maps activities to bodyposition levels
humanmotion_general = {
    "sitting: in car\/bus\/tram\/train": ["sitting"],
    "sitting: somewhere else": ["sitting"],
    "sitting: on the couch": ["sitting"],
    "sitting: at a table": ["sitting"],
    "sitting: somewhere else\",\"Other (please specify)": ["sitting"],
    "sitting: somewhere else\",\"Other: sitting on bed": ["sitting"],
    "sitting: on the couch\",\"sitting: somewhere else": ["sitting"],
    "sitting: at a table\",\"sitting: in car\/bus\/tram\/train": ["sitting"],

    "standing": ["standing"],
    "standing\",\"Other (please specify)": ["standing"],
    "standing\",\"Other: in kitchen cooking": ["standing"],
    "standing\",\"Other: in train standing": ["standing"],

    "walking": ["walking"],
    "Other: walking": ["walking"],

    "running": ["running"],
    "cycling": ["cycling"],
    "lying": ["lying"],
    "lying\",\"Other: auf der Shakti matte": ["lying"]
}

humanmotion_specific = {
    "lying": ["lying"],
    "lying\",\"Other: auf der Shakti matte": ["lying"],

    "sitting: at a table": ["sitting at a table"],
    "sitting: at a table\",\"sitting: in car\/bus\/tram\/train": ["sitting at a table"],

    "sitting: on the couch": ["sitting on the couch"],
    "sitting: on the couch\",\"sitting: somewhere else": ["sitting on the couch"],

    "sitting: in car\/bus\/tram\/train": ["sitting in car/bus/tram/train"],

    "sitting: somewhere else": ["sitting somewhere else"],

    "standing": ["standing"],
    "walking": ["walking"],
    "running": ["running"],
    "cycling": ["cycling"],

}



# create new column with general human motion
## create new column "humanmotion_general" which contains values of "bodyposition" if they are in the dictionary
df_esm["humanmotion_general"] = df_esm.apply(lambda x : x["bodyposition"] if (x["bodyposition"] in  humanmotion_general) else np.nan, axis = 1)
df_esm["humanmotion_general"] = df_esm["humanmotion_general"].replace(humanmotion_general)
df_esm["humanmotion_general"].value_counts()
df_esm["humanmotion_general"].isna().sum()

## create new column with specific human motion and leave every cell which is not in the dictionary as NaN
df_esm["humanmotion_specific"] = df_esm.apply(lambda x : x["bodyposition"] if x["bodyposition"] in  humanmotion_specific.keys() else np.nan, axis = 1)
df_esm["humanmotion_specific"] = df_esm["humanmotion_specific"].replace(humanmotion_specific)
df_esm["humanmotion_specific"].value_counts()
df_esm["humanmotion_specific"].isna().sum()
#endregion

#region before & after sleep
before_after_sleep = {
    "lying in bed after sleeping": ["lying in bed after sleeping"],

    "lying in bed before sleeping": ["lying in bed before sleeping"],

    "lying in bed at another time": ["lying in bed at other times"],

    "lying on couch": ["lying on the couch"],

}

#endregion

#region on the toilet
on_the_toilet_sittingsomewhereelse = {
    "sitting: on the couch": ["sitting (not on the toilet)"],
    "sitting: at a table": ["sitting (not on the toilet)"],
}

on_the_toilet = {
    "on the toilet": ["on the toilet"]
}
#endregion

# create dataframe with details of new columns
new_columns = [["human motion - general", "humanmotion_general", "bodyposition"],
               ["human motion - specific", "humanmotion_specific", "bodyposition"],
               ["before & after sleep", "before_after_sleep", "activity"],
               ["on the toilet", "on_the_toilet_sittingsomewhereelse", "bodyposition"],
               ["on the toilet", "on_the_toilet", "activity"]]

df_new_columns = pd.DataFrame(new_columns, columns = ["column name", "column dictionary", "column to map"])

# iterate through dataframe and create new columns
for index, row in df_new_columns.iterrows():
    # check if column_name already exists; if it does, then just add the data frome the current column dictionary (not replace it)
    # this is relevant i.e. for "on the toilet" as it is a combination of two columns
    if row["column name"] not in df_esm.columns:
        # create new column
        df_esm[row["column name"]] = df_esm.apply(lambda x : x[row["column to map"]] if (x[row["column to map"]] in  eval(row["column dictionary"])) else np.nan, axis = 1)
        # replace values in new column
        df_esm[row["column name"]] = df_esm[row["column name"]].replace(eval(row["column dictionary"]))

    else:
        # add data from current column dictionary to existing column
        df_esm[row["column name"]] = df_esm.apply(lambda x : x[row["column to map"]] if (x[row["column to map"]] in  eval(row["column dictionary"])) else x[row["column name"]], axis = 1)
        print("column already exists: " + row["column name"])


