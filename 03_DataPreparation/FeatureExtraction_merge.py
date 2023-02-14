#region import
import pandas as pd


#endregion

# create object to merge low & high-frequency features
class Features_Merge:
    def __init__(self, df_features_highfreq, features_highfreq_lensegment, df_features_lowfreq):
        self.df_features_highfreq = df_features_highfreq
        self.features_highfreq_lensegment = features_highfreq_lensegment
        self.df_features_lowfreq = df_features_lowfreq


    def merge(self):

#testdata
## load pkl
df_features_highfreq = pd.read_pickle("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/activity-label_human motion - general_highfrequencysensors-all_timeperiod-2 s_featureselection.pkl")
features_highfreq_lensegment = 2 #seconds
df_features_lowfreq = pd.read_csv("/Users/benediktjordan/Documents/MTS/Iteration01/Data/data_preparation/features/locations/locations-aroundevents_features-distance-speed-acceleration.csv")