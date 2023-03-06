#region import
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import time
import matplotlib
import seaborn as sns
from geopy.distance import geodesic

import utm
from collections import Counter
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

#endregion

# this class contains the method to generate a generic map with GPS points
class GPS_visualization:

    # visualize GPS points on generic map (include scale bar)
    def gps_utm_genericmap(df, figure_title, label_column_name = None, colour_based_on_labels = "no"):

        # convert column names: if column name contains 'latitude' or 'longitude', replace it with 'latitude' or 'longitude'
        for column in df.columns:
            if 'latitude' in column:
                df = df.rename(columns={column:'latitude'})
            elif 'longitude' in column:
                df = df.rename(columns={column:'longitude'})


        # convert latitude and longitude to UTM
        x, y, _, _ = zip(*df.apply(lambda row: utm.from_latlon(row['latitude'], row['longitude']), axis=1))

        # count the occurrences of each point: necessary to plot the points with the correct size
        c = Counter(zip(x, y))
        print("The Maximum number of points at one location is: ", max(c.values()))
        # create a list of the sizes, here multiplied by 10 for scale
        ## scale this point scaller depending on how many points are in the biggest cluster
        scale_factor = 50
        if max(c.values()) > 100:
            scale_factor = 1
        if max(c.values()) > 30000:
            scale_factor = 0.8
        if max(c.values()) > 50000:
            scale_factor = 0.1
        if max(c.values()) > 100000:
            scale_factor = 0.1
            if max(c.values()) > 200000:
                scale_factor = 0.07
        if max(c.values()) > 300000:
            scale_factor = 0.05
        s = [scale_factor * c[(xx, yy)] for xx, yy in zip(x, y)]

        # visualize
        fig, ax = plt.subplots(figsize= (15,10))

        #create sns scatterplot
        if colour_based_on_labels == "yes":
            # create colour palette
            ## Get the "bright" palette
            pal = sns.color_palette("bright")
            hex_colors = [matplotlib.colors.rgb2hex(color) for color in pal]
            ## remove similar colors from the palette
            hex_colors = [color for color in hex_colors if
                          color != '#023eff' and color != '#e8000b' and color != '#f14cc1' and color != '#1ac938']
            hex_colors = hex_colors[:len(df[label_column_name].unique())]

            #plot
            sns.scatterplot(x, y, hue = df[label_column_name], s=s, ax=ax, palette=hex_colors)
        else:
            sns.scatterplot(x, y, s=s, ax=ax)
        #ax.scatter(x, y, s=s)
        ax.set_aspect('equal')  # set x and y-axis scales to be equal

        # dont show numbers on x and y-axis
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ## calculate lenght of scalebar which should be 10% of the x-axis
        scalebar_length = (max(x) - min(x)) * 0.1
        scalebar_title = str(round(float(scalebar_length / 1000), 1)) + ' km'

        # add a scale bar with AnchoredSizeBar
        scalebar = AnchoredSizeBar(ax.transData,
                                   scalebar_length, scalebar_title, 'lower right',
                                   pad=0.5,
                                   color='black',
                                   frameon=False,
                                   size_vertical=1)

        ax.add_artist(scalebar)

        #add title
        plt.title(figure_title, fontsize=18)
        plt.tight_layout()
        plt.show()
        return fig








#region OUTDATED: first tries to visualize GPS data in generic maps

##get complete locations dataset of all participants
df_locations = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/datasets/locations_all.pkl")
df_locations = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/datasets/locations_all.csv")

df_locations_events = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/xmin_around_events/locations_esm_timeperiod_5 min.csv_JSONconverted.pkl")

# label locations with event
sensors_included = "all"
with open("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_" + sensors_included + "_transformed_labeled_dict.pkl",
        'rb') as f:
    dict_label = pickle.load(f)
label_column_name = "label_before and after sleep"
df_locations_events = labeling_sensor_df(df_locations_events, dict_label, label_column_name, "ESM_timestamp")
# delete all rows without label
df_locations_events = df_locations_events[df_locations_events[label_column_name].notna()]
# check how many events are there
print("Number of events: ", len(df_locations_events["ESM_timestamp"].unique()))
print("Number of participants with events: ", len(df_locations_events["loc_device_id"].unique()))
# only count one event per unique ESM_timestamp and how the label is distributed
df_locations_events_vis = df_locations_events.drop_duplicates(subset=["ESM_timestamp"])
print(df_locations_events_vis[label_column_name].value_counts())
print("Number of labels per participant: ", df_locations_events_vis.groupby("loc_device_id")[label_column_name].value_counts())


# only fist 1000 rows
df_locations_part = df_locations[:1000]
# visualize GPS data
## create class which a) creates a map with all GPS coordinates and b) creates a scatterplot for every participant and c) creates a map with locations at sleep of participants (if available)
class Visualize_GPS:
    def __init__(self, df_locations, df_locations_events, label_column_name):
        self.df_locations = df_locations
        self.df_locations_events = df_locations_events
        self.label_column_name = label_column_name

    # create scatterplot for every participant
    def create_scatterplot_participant(self):
        # create scatterplot for every participant
        for participant in self.df_locations["device_id"].unique():
            plt.figure(figsize=(20, 10))
            plt.scatter(self.df_locations[self.df_locations["device_id"] == participant]["double_longitude"],
                        self.df_locations[self.df_locations["device_id"] == participant]["double_latitude"], s=0.1)
            plt.title("Map of GPS coordinates of participant " + str(participant))
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.show()

    # create map with locations at sleep of participants (if available)
    def create_map_sleep(self):
        for participant in self.df_locations_events["loc_device_id"].unique():
            plt.figure(figsize=(20, 10))
            # create scatterplot and colour points according to label using seaborn
            sns.scatterplot(x=self.df_locations_events[self.df_locations_events["loc_device_id"] == participant]["loc_double_latitude"],
                            y=self.df_locations_events[self.df_locations_events["loc_device_id"] == participant]["loc_double_longitude"],
                            hue=self.df_locations_events[self.df_locations_events["loc_device_id"] == participant][self.label_column_name])
            plt.title("Map of GPS coordinates of participant " + str(participant))
            # include subtitle
            plt.suptitle("Number of events: " + str(len(self.df_locations_events[self.df_locations_events["loc_device_id"] == participant]["ESM_timestamp"].unique())), fontsize=16)
            plt.xlabel("Latitude")
            plt.ylabel("Longitude")
            plt.show()

# create instance of class
visualize_gps = Visualize_GPS(df_locations, df_locations_events, label_column_name)
visualize_gps.create_map_sleep()

# basic stats of GPS data
df_locations_events["loc_double_latitude"].describe()
df_locations_events["loc_double_longitude"].describe()
## find out how many locations participants are sleeping at

#region create distance legend for map



#temporary: delete all ESM related columns excelt ESM_timestamp
## checkout labeling
df_esm = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/esm_all_transformed_labeled.csv")
label_columns = [col for col in df_esm.columns if "label" in col]

sensors_included = "all"
with open("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/labels/esm_" + sensors_included + "_transformed_labeled_dict.pkl",
        'rb') as f:
    dict_label = pickle.load(f)
label_column_name = "label_human motion - general"
df_locations_new = labeling_sensor_df(df_locations_events, dict_label, label_column_name, ESM_identifier_column = "ESM_timestamp")



#region temporary plot GPS data on worldmap
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame

# use only first 1000 rows
#df_locations_events = df_locations_events[:10000]
geometry = [Point(xy) for xy in zip(df_locations_events["loc_double_longitude"], df_locations_events["loc_double_latitude"])]
gdf = GeoDataFrame(df_locations_events[["loc_double_latitude", "loc_double_longitude"]], geometry=geometry)

#this is a simple map that goes with geopandas
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
image = gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', color='red', markersize=15)
image.figure.savefig("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_exploration/GPS_visualization/Map.png")
#endregion

#region temporary: delete the ESM related columns (except ESM_timestamp) in the x_min around events datasets
for root, dirs, files in os.walk("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/xmin_around_events"):
    for file in files:
        if not file.endswith(".csv"):
            print("skipped file: " + str(file))
            continue
        # load csv
        df = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/xmin_around_events/" + file)
        # delete all ESM columns except column "ESM_timestamp"
        df = df.drop(columns=["ESM_location", "ESM_location_time",
                     "ESM_bodyposition", "ESM_bodyposition_time",
                     "ESM_activity", "ESM_activity_time",
                     "ESM_smartphonelocation", "ESM_smartphonelocation_time",
                     "ESM_aligned", "ESM_aligned_time"])
        # save as pickle
        df.to_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/xmin_around_events/" + file[:-4] + ".pkl")
        print("file saved: ", file)
#endregion

#endregion

#endregion



