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

# region human motion
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
#endregion

# region public transport
activity_list = df_esm["bodyposition"].value_counts()
location_list = df_esm["location"].value_counts()

publictransport = {
    "on the way: in public transport": ["public transport"],

    "on the way: in train": ["train"],

    "on the way: in car": ["car"],

    "on the way: walking\/cycling": ["walking/cycling"],

    "on the way: other means of transport": ["other means of transport"],

    "at home": ["somewhere else (home, office, other workplace, etc.)"],
    "in home-office": ["somewhere else (home, office, other workplace, etc.)"],
    "in the office": ["somewhere else (home, office, other workplace, etc.)"],
    "at another workplace": ["somewhere else (home, office, other workplace, etc.)"],
    "on the way: standing": ["somewhere else (home, office, other workplace, etc.)"],
    "in restaurant": ["somewhere else (home, office, other workplace, etc.)"]
}

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

#region during work
df_esm["activity"].value_counts()
location = {
    "in the office": ["in the office"],

    "at another workplace": ["at another workplace"],

    "in home-office": ["at home"],
    "at home": ["at home"],

    "at a friends place": ["at a friends place"],

    "in restaurant": ["in restaurant"],

    "shopping": ["shopping"],

    "on the way: in train": ["on the way"],
    "on the way: walking/\cycling": ["on the way"],
    "on the way: in car": ["on the way"],
    "on the way: standing": ["on the way"],
    "on the way: in public transport": ["on the way"],

    "with friends outside": ["with friends outside"]
}
#endregion

# region smartphonelocation
df_esm["smartphonelocation"].value_counts()
smartphonelocation = {
    "in one hand": ["in one hand"],
    "in two hands": ["in two hands"],
    "on a flat surface in front of me (i.e. table, bed, etc.)": ["on flat surface"],
    "in smartphone mount (i.e. on bicycle, in car, etc.)" : ["in smartphone mount"]
}
#endregion

# region aligned
df_esm["aligned"].value_counts()
aligned = {
    "Aligned with my intentions": ["aligned"],
    "Not aligned with my intentions": ["not aligned"]
}
#endregion


# create dataframe with details of new columns
new_columns = [["label_human motion - general", "humanmotion_general", "bodyposition"],
               ["label_human motion - specific", "humanmotion_specific", "bodyposition"],
               ["label_before and after sleep", "before_after_sleep", "activity"],
               ["label_on the toilet", "on_the_toilet_sittingsomewhereelse", "bodyposition"],
               ["label_on the toilet", "on_the_toilet", "activity"],
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
        # replace values in new column
        df_esm[row["column name"]] = df_esm[row["column name"]].replace(eval(row["column dictionary"]))

    else:
        # add data from current column dictionary to existing column
        df_esm[row["column name"]] = df_esm.apply(lambda x : x[row["column to map"]] if (x[row["column to map"]] in  eval(row["column dictionary"])) else x[row["column name"]], axis = 1)
        print("column already exists: " + row["column name"])


df_esm.to_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/esm_all_transformed_labeled.csv")
df_esm = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/esm_all_transformed_labeled.csv")

# create timestamp_datetime column
df_esm["timestamp_datetime"] = pd.to_datetime(df_esm["timestamp"], unit = "ms")

# convert df_esm into dict (set timestamp as keys) and save it
df_esm = df_esm.set_index("timestamp_datetime")
df_esm_dict = df_esm.to_dict(orient = "index")
with open("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_all_transformed_labeled_dict.pkl", 'wb') as f:
    pickle.dump(df_esm_dict, f, pickle.HIGHEST_PROTOCOL)

# check if there are duplicates in timestamp
test = df_esm["timestamp"].value_counts()
test.describe()
