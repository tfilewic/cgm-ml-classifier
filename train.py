#tfilewic 2025-09-10

import sys
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pickle

CGM1_PATH = "CGMData.csv"
CGM2_PATH = "CGM_patient2.csv"
INSULIN1_PATH = "InsulinData.csv"
INSULIN2_PATH = "Insulin_patient2.csv"
MODEL_PATH = "model.pkl"

INSULIN_FEATURES = ["Timestamp", "BWZ Carb Input (grams)"]
CGM_FEATURES = ["Timestamp", "Sensor Glucose (mg/dL)"]
EXTRACTED_FEATURES = ["ttp", "normalized_difference", "fft", "range", "max_d1_3", "max_d2", "quarter_slope"]


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
        if pd.isna(next_meal): 
            continue
            
        start = this_meal + pd.Timedelta("2h")
        end = next_meal - pd.Timedelta("2h")

        while (start < end):
            nomeals.append(start)
            start += pd.Timedelta("2h")

    return pd.DataFrame({"Timestamp": nomeals})


def build_meal_matrix(meals: pd.DataFrame, cgm: pd.DataFrame) -> np.array:
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

    return np.array(matrix, dtype=float)

def build_nomeal_matrix(nomeals: pd.DataFrame, cgm: pd.DataFrame) -> np.array:
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

    return np.array(matrix, dtype=float)

def create_feature_row(glucose: np.ndarray) -> np.ndarray:
    """
    creates a row of features from a period of glucose readings
    """
    n = len(glucose)
    minimum = float(glucose.min())
    maximum = float(glucose.max())

    quarter = n // 4
    quarter_slope = (glucose[quarter+1] - glucose[quarter-1]) / 2.0

    smoothed3 = np.convolve(glucose, np.array([1,1,1])/3.0, mode="same")
    start = n // 5  #ignore first 20%
    end = n - 2     #ignore trailing edge
    peak_index = start + int(np.argmax(smoothed3[start:end]))
    ttp = peak_index / (n - 1)

    normalized_difference = (maximum - minimum) / minimum
    range = maximum - minimum
    
    d1_3 = glucose[2:] - glucose[:-2]  #across 3pts
    max_d1_3 = np.max(np.abs(d1_3))

    d2 = np.diff(np.diff(glucose))
    max_d2 = np.max(np.abs(d2))
    
    fft_vals = np.fft.fft(glucose)
    power = np.abs(fft_vals) ** 2
    fft = power[1]

    return np.array(
        [ttp, normalized_difference, fft, range, max_d1_3, max_d2, quarter_slope],
        dtype=float
    )

def extract_features(matrix: np.ndarray) -> np.ndarray:
    """
    extracts features from each row in a meal or nomeal matrix
    """
    features = [create_feature_row(row) for row in matrix]
    return np.vstack(features)



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
meal_matrix = np.vstack([build_meal_matrix(meals1, cgm1), build_meal_matrix(meals2, cgm2)])
nomeal_matrix = np.vstack([build_nomeal_matrix(nomeals1, cgm1), build_nomeal_matrix(nomeals2, cgm2)])


''' FEATURE EXTRACTION '''
meal_features = extract_features(meal_matrix)
nomeal_features = extract_features(nomeal_matrix)


''' TRAIN MODEL '''
X = np.vstack([meal_features, nomeal_features])
y = np.hstack([np.ones(len(meal_features)), np.zeros(len(nomeal_features))])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"))
clf.fit(X, y)

with open(MODEL_PATH, "wb") as output_file:
    pickle.dump(clf, output_file)

