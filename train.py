#tfilewic 2025-09-09

import sys
import pandas as pd
import numpy as np

CGM1_PATH = "CGMData.csv"
CGM2_PATH = "CGM_patient2.csv"
INSULIN1_PATH = "InsulinData.csv"
INSULIN2_PATH = "Insulin_patient2.csv"

INSULIN_FEATURES = ["Timestamp", "BWZ Carb Input (grams)"]
CGM_FEATURES = ["Timestamp", "Sensor Glucose (mg/dL)"]

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

    #fix index
    meals.reset_index(drop=True, inplace=True)

    return meals

def get_nomeals(df: pd.DataFrame) -> pd.DataFrame:
    """
    calculates start times of eligible postabsorptive windows
    """
    meals = df.copy()

    #drop NaNs
    meals.dropna(subset=["BWZ Carb Input (grams)"], inplace=True)

    #drop 0s
    meals.drop(meals[meals["BWZ Carb Input (grams)"] == 0].index, inplace=True)

    nomeals = []
    for this_meal, next_meal in zip(meals["Timestamp"], meals["Timestamp"].shift(1)):
        
        #skip first row edge case
        if pd.isna(next): 
            continue
            
        start = this_meal + pd.Timedelta("2h")
        end = next_meal - pd.Timedelta("2h")

        while (start < end):
            nomeals.append(start)
            start += pd.Timedelta("2h")

    return pd.DataFrame({"Timestamp": nomeals})


def build_meal_matrix(meals: pd.DataFrame, cgm: pd.DataFrame) -> pd.DataFrame:
    """
    builds 30 sample absorptive windows from meal times 
    """
    THRESHOLD = 2
    matrix = []

    for meal in meals["Timestamp"]:
        window = cgm[(cgm["Timestamp"] >= meal - pd.Timedelta("30min")) &
                     (cgm["Timestamp"] <= meal + pd.Timedelta("2h"))]

        values = window["Sensor Glucose (mg/dL)"].to_numpy()

        #skip windows that dont have 30 pts
        if len(values) != 30:
            continue
        
        #discard if missing data points exceed threshold
        missing = np.isnan(values).sum()
        if missing > THRESHOLD:
            continue
        
        #fill missing
        if missing > 0:
            series = pd.Series(values)
            values = series.interpolate(limit_direction="both").to_numpy()

        #insert row
        matrix.append(values)

    return pd.DataFrame(matrix)

def build_nomeal_matrix(nomeals: pd.DataFrame, cgm: pd.DataFrame) -> pd.DataFrame:
    """
    builds 24 sample postabsorptive windows from nomeal times 
    """
    THRESHOLD = 2
    matrix = []

    for nomeal in nomeals["Timestamp"]:
        window = cgm[(cgm["Timestamp"] >= nomeal) &
                     (cgm["Timestamp"] <= nomeal + pd.Timedelta("2h"))]

        values = window["Sensor Glucose (mg/dL)"].to_numpy()

        #skip windows that dont have 24 pts
        if len(values) != 24:
            continue
        
        #discard if missing data points exceed threshold
        missing = np.isnan(values).sum()
        if missing > THRESHOLD:
            continue
        
        #fill missing
        if missing > 0:
            series = pd.Series(values)
            values = series.interpolate(limit_direction="both").to_numpy()

        #insert row
        matrix.append(values)

    return pd.DataFrame(matrix)



''' PREPROCESSING '''

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

#discard unused columns
select_features(INSULIN_FEATURES, insulin1)
select_features(INSULIN_FEATURES, insulin2)
select_features(CGM_FEATURES, cgm1)
select_features(CGM_FEATURES, cgm2)

#calculate window start times
meals1 = get_meals(insulin1)
meals2 = get_meals(insulin2)
nomeals1 = get_nomeals(insulin1)
nomeals2 = get_nomeals(insulin2)

#create matrices
meal_matrix = pd.concat([build_meal_matrix(meals1, cgm1), build_meal_matrix(meals2, cgm2)], ignore_index=True)
nomeal_matrix = pd.concat([build_nomeal_matrix(meals1, cgm1), build_nomeal_matrix(meals2, cgm2)], ignore_index=True)


''' FEATURE EXTRACTION '''
 
