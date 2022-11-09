#region import
import pandas as pd
#endregion

# create clustering object
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
from sklearn.cluster import KMeans

class Clustering:
    def __init__(self, df):
        self.df = df
