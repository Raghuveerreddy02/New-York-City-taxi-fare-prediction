# %% [code]
import numpy as np 
import pandas as pd
import scipy as scipy
import datetime as dt
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
import os
import gc

# %% [markdown]
# # Reading The Data

# %% [code]
train =  pd.read_csv('../input/train.csv', nrows = 24200000)

# %% [markdown]
# # Data Cleaning

# %% [markdown]
# Drop rows with null value

# %% [code]
print("Total Rows before data cleaning: ", train.shape[0])

# %% [code]
train = train.dropna(how = 'any', axis = 'rows')

# %% [markdown]
# Remove rows where:
# * Fare Amount is less than 0
# * Fare Amount is more than $400
# * Pickup & Dropoff Latitude & Longitude are 0

# %% [code]
train = train.loc[ (train.fare_amount > 0)  & (train.fare_amount <= 300) & (train.pickup_longitude != 0) & (train.pickup_latitude != 0) & (train.dropoff_longitude != 0) & (train.dropoff_latitude != 0)]

# %% [code]
print("Total Rows after data cleaning: ", train.shape[0])

# %% [markdown]
# # Define A Function, that calculates Distance using Haversine Distance Formula and calculates Bearing
# 
# https://en.wikipedia.org/wiki/Haversine_formula
# https://en.wikipedia.org/wiki/Bearing_(navigation)

# %% [code]
def havesine_distance_and_bearing(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude,getBearing):

    #Define earth radius (km)
    R_earth = 6371
    #Convert degrees to radians
    pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude = map(np.radians, [pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude])
    
    #Compute distances along lat, lon dimensions
    dlatitude = dropoff_latitude - pickup_latitude
    dlongitude = dropoff_longitude - pickup_longitude
    
    #Compute haversine distance
    harversineDistance = np.sin(dlatitude/2.0)**2 + np.cos(pickup_latitude) * np.cos(dropoff_latitude) * np.sin(dlongitude/2.0)**2
    
    
    if getBearing:
        #Compute Bearing Distance
        bearing = np.arctan2(np.sin(dlongitude * np.cos(dropoff_latitude)),np.cos(pickup_latitude) * np.sin(dropoff_latitude) - np.sin(pickup_latitude) * np.cos(dropoff_latitude) * np.cos(dlongitude))

        return 2 * R_earth * np.arcsin(np.sqrt(harversineDistance)), bearing
    else:
        return 2 * R_earth * np.arcsin(np.sqrt(harversineDistance))

# %% [markdown]
# # Add Distance From Airport

# %% [code]
def add_airport_dist(df):
    """
    Return minumum distance from pickup or dropoff coordinates to each airport.
    NYC: Newyork Central
    SOL: Statue of Liberty 
    JFK: John F. Kennedy International Airport
    LGA: LaGuardia Airport
    EWR: Newark Liberty International Airport
    """
    nyc_coord = (40.7141667,-74.0063889) 
    sol_coord = (40.6892,-74.0445)
    jfk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    
    
    pickup_lat = df['pickup_latitude']
    dropoff_lat = df['dropoff_latitude']
    pickup_lon = df['pickup_longitude']
    dropoff_lon = df['dropoff_longitude']
    
    pickup_jfk = havesine_distance_and_bearing(pickup_lat, pickup_lon, jfk_coord[0], jfk_coord[1],False) 
    dropoff_jfk = havesine_distance_and_bearing(jfk_coord[0], jfk_coord[1], dropoff_lat, dropoff_lon,False) 
    pickup_ewr = havesine_distance_and_bearing(pickup_lat, pickup_lon, ewr_coord[0], ewr_coord[1],False)
    dropoff_ewr = havesine_distance_and_bearing(ewr_coord[0], ewr_coord[1], dropoff_lat, dropoff_lon,False) 
    pickup_lga = havesine_distance_and_bearing(pickup_lat, pickup_lon, lga_coord[0], lga_coord[1],False) 
    dropoff_lga = havesine_distance_and_bearing(lga_coord[0], lga_coord[1], dropoff_lat, dropoff_lon,False)
    pickup_sol = havesine_distance_and_bearing(pickup_lat, pickup_lon, sol_coord[0], sol_coord[1],False) 
    dropoff_sol = havesine_distance_and_bearing(sol_coord[0], sol_coord[1], dropoff_lat, dropoff_lon,False)
    pickup_nyc = havesine_distance_and_bearing(pickup_lat, pickup_lon, nyc_coord[0], nyc_coord[1],False) 
    dropoff_nyc = havesine_distance_and_bearing(nyc_coord[0], nyc_coord[1], dropoff_lat, dropoff_lon,False)
    

    df['nyc_dist'] = pickup_nyc + dropoff_nyc
    df['jfk_dist'] = pickup_jfk + dropoff_jfk
    df['ewr_dist'] = pickup_ewr + dropoff_ewr
    df['lga_dist'] = pickup_lga + dropoff_lga
    df['sol_dist'] = pickup_sol + dropoff_sol
    
    return df

# %% [markdown]
# # Define Function for getting Date, Month, Year, Weekday, Hour from date column

# %% [code]
def get_datetimeinfo(df):
    #Convert to datetime format
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'],format="%Y-%m-%d %H:%M:%S UTC")
    
    df['day'] = df.pickup_datetime.dt.day
    df['month'] = df.pickup_datetime.dt.month
    df['year'] = df.pickup_datetime.dt.year
    df['weekday'] = df.pickup_datetime.dt.weekday
    df['hour'] = df.pickup_datetime.dt.hour

    return df

# %% [markdown]
# # Apply all the above functions to the data

# %% [code]
train = get_datetimeinfo(train)

# %% [code]
## Remove Unwanted Columns
train.drop(columns=['key', 'pickup_datetime'], inplace=True)
gc.collect()

# %% [code]
train['distance'], train['bearing'] = havesine_distance_and_bearing(train['pickup_latitude'], train['pickup_longitude'], train['dropoff_latitude'] , train['dropoff_longitude'],True) 

# %% [code]
train = add_airport_dist(train)

# %% [markdown]
# # Our Table after feature engineering

# %% [code]
train.head()

# %% [markdown]
# # Get Target Variable in seperate column

# %% [code]
y = train['fare_amount']
train = train.drop(columns=['fare_amount'])

# %% [markdown]
# # Split the data into train & validation set
train = np.array(train)
y = np.array(y)
gc.collect()
# %% [code]
train,valid,y,y_valid = train_test_split(train,y,random_state=123,test_size=0.12)
gc.collect()

# %% [markdown]
# # Apply Linear Regression Algorithm

# %% [code]
model = CatBoostRegressor(iterations=50000,
                             learning_rate=0.5,
                             eval_metric='RMSE',
                             random_seed = 5,
                             early_stopping_rounds=200
                            )
model.fit(train, y,
             eval_set=(valid, y_valid),
             use_best_model=True,
             verbose=False)

# %% [markdown]
# # Read Test.CSV

# %% [code]
test =  pd.read_csv('../input/test.csv')

# %% [markdown]
# # Perform Feature Engineering on Test.CSV

# %% [code]
test = get_datetimeinfo(test)
test['distance'], test['bearing'] = havesine_distance_and_bearing(test['pickup_latitude'], test['pickup_longitude'], test['dropoff_latitude'] , test['dropoff_longitude'],True) 
test = add_airport_dist(test)

# %% [code]
## Remove Unwanted Columns
test_key = test['key']
test.drop(columns=['key', 'pickup_datetime'], inplace=True)

# %% [code]
test = test.replace([np.inf, np.NINF,np.nan], 0)

# %% [markdown]
# # Predict on test dataset

# %% [code]
test['fare_amount'] = model.predict(test)      

# %% [markdown]
# # Save CSV File, so we can submit it to competition

# %% [code]
test['key'] = test_key
test[['key','fare_amount']].to_csv('submission.csv',index=False)