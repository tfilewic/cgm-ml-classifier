#tfilewic 2025-09-07

import sys
import pandas as pd

CGM1_PATH = "CGMData.csv"
CGM2_PATH = "CGM_patient2.csv"
INSULIN1_PATH = "InsulinData.csv"
INSULIN2_PATH = "Insulin_patient2.csv"

INSULIN_FEATURES = ["Timestamp", "BWZ Carb Input (grams)"]

def import_file(filename:str) -> pd.DataFrame:
    """
    reads csv into dataframe
    """
    print(f"Importing {filename} ...")
    try:
        return pd.read_csv(filepath_or_buffer=filename, header=0, low_memory=False)   #read data
    except Exception as e:
        sys.exit(f"ERROR: Failed to import {filename}: {e}")

def create_timestamps(df: pd.DataFrame) -> None:
    """
    derives Timestamp feature 
    from combines Date and Time columns 
    """
    date = df["Date"]
    time = df["Time"]
    timestamp = pd.to_datetime(date + " " + time, format='mixed')
    df.insert(0, "Timestamp", timestamp)

def select_features(features: list[str], df: pd.DataFrame) -> None:
    """
    removes features that are not listed in the features param
    """
    df.drop(columns=[column for column in df.columns if column not in features], inplace=True)

def get_meals(df: pd.DataFrame) -> pd.DataFrame:
    """
    keeps only start times of eligible meals
    """
    meals = df.copy()

    #drop NaNs
    meals.dropna(subset=["BWZ Carb Input (grams)"], inplace=True)

    #drop 0s
    meals.drop(meals[meals["BWZ Carb Input (grams)"] == 0].index, inplace=True)

    #drop meals which are followed by another meal within 2 hours
    too_soon = (meals["Timestamp"].shift(1) - meals["Timestamp"] <= pd.Timedelta("2h"))
    meals.drop(meals.index[too_soon], inplace=True)

    #drop carb input column
    meals.drop(columns=["BWZ Carb Input (grams)"], inplace=True)

    return meals

def get_postabsorptive(df: pd.DataFrame) -> pd.DataFrame:
    """
    calculates start times of eligible postabsorptive windows
    """

    #drop meals which are followed by another meal within 4 hours
    interrupted = (df["Timestamp"].shift(1) - df["Timestamp"] <= pd.Timedelta("4h"))
    meals = df.drop(df.index[interrupted])

    #calculate start time of absorptive window from meal timestamp
    postabsorptive = meals["Timestamp"] + pd.Timedelta("2h") 

    return postabsorptive

'''
Extraction: Meal data
The start of a meal can be obtained from InsulinData.csv. 
Search column Y for a non-NAN non-zero value. This time indicates the start of a meal. There can be three conditions:
    1. There is no meal from time tm to time tm+2hrs. Then use this stretch as meal data.
    2. There is a meal at some time tp in between tp>tm and tp< tm+2hrs. Ignore the meal data at time tm and consider the meal at time tp instead.
    3. There is a meal at time tm+2hrs, then consider the stretch from tm+1hr 30min to tm+4hrs as meal data.
'''
#import insulin data
insulin1 = import_file(INSULIN1_PATH)
insulin2 = import_file(INSULIN2_PATH)
cgm1 = import_file(CGM1_PATH)
cgm2 = import_file(CGM2_PATH)

#create timestamps
create_timestamps(insulin1)
create_timestamps(insulin2)
create_timestamps(cgm1)
create_timestamps(cgm2)

#discard all but timestamps and col Y "BWZ Carb Input (grams)"
select_features(INSULIN_FEATURES, insulin1)
select_features(INSULIN_FEATURES, insulin2)

#filter meal start times
meals1 = get_meals(insulin1)
meals2 = get_meals(insulin2)

#calculate no-meal start times
postabsorptive1 = get_postabsorptive(meals1)
postabsorptive2 = get_postabsorptive(meals2)





 
#debug
print(meals1)
print(postabsorptive1)
meals1.to_csv("mmmmmm.csv")
postabsorptive1.to_csv("ppppppp.csv")


'''
Extraction: No Meal data 
The start of no meal is at time tm+2hrs where tm is the start of some meal. 
We need to obtain a 2hr stretch of no meal time. 
So you need to find all 2 hr stretches in a day that have no meal and do not fall within 2 hrs of the start of a meal.
'''



'''
Handling missing data: 
You have to carefully handle missing data. This is an important data mining step that is required for many applications. 
Here there are several approaches: 
    1. Ignore the meal or no meal data stretch if the number of missing data points in that stretch is greater than a certain threshold.
    2. Use linear interpolation (not a good idea for meal data but maybe for no meal data).
    3. Use polynomial regression to fill up missing data (untested in this domain). 
Choose wisely.
'''



'''
Feature Extraction and Selection: 
You have to carefully select features from the meal time series that are discriminatory between meal and no meal classes.
'''