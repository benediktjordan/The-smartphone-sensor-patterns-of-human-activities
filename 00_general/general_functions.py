# Note:
# SKLEARN: upgraded sklearn on 9. February from 1.1.2. to (didnt work out!)
# in order to do so also had to upgrade pi from  22.3.1 -> 23.0


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



#calculate figure width for matplotlib so it conforms with Latex based on this tutorial: https://jwalton.info/Embed-Publication-Matplotlib-Latex/
# Note: something doesnt quite work, figures still in Latex much smaller than page. Check current Latex settings
def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim
width = 418.25555 # this is the width of my Latex document
fig_dim = set_size(width, fraction=1)

#endregion