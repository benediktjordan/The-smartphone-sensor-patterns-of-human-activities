#region calculate new features

class computeFeatures:
    # Class Variable
    animal = 'dog'

    # The init method or constructor
    def __init__(self, breed, color):
        # Instance Variable
        self.breed = breed
        self.color = color


    # for battery_charges
    ## calculate difference between "before bed" event/"after bed" event timestamps and "charge battery" event timestamp/"charge
    ## battery double_end_timestamp" timestamp

    def battery_charges_features(self, df):
        # calculate difference between "before bed" event/"after bed" event timestamps and "charge battery" event timestamp/"charge
        # battery double_end_timestamp" timestamp

        return df

