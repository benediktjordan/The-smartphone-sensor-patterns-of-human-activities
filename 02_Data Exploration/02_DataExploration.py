# region import
import numpy as np
import os
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import json
#from datetime import *


# endregion


#region visualizations




#endregion






# region get all different users & delete the ones, which I already know
esm_all_users = esm_all
for index_users, user in users.iterrows():
    esm_all_users = esm_all_users[esm_all_users["2"] != user["ID"]]
    print("Dropped user: " + str(user["Name"]))
esm_all_users["2"].unique()

# endregion


# region labels: join all aware_device files
dir_databases = "/Volumes/TOSHIBA EXT/InUsage/Databases"
sensor = "aware_device"
aware_device_all = join_sensor_files(dir_databases, sensor)

aware_device_all.to_csv("/Volumes/TOSHIBA EXT/InUsage/Databases/aware_device_all.csv")

#endregion
