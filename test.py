#tfilewic 2025-09-10

import pickle
import pandas as pd
import numpy as np
from train import create_feature_row

RESULTS_PATH = "Result.csv"

model = pickle.load(open("model.pkl", "rb"))    #load model
test_matrix = pd.read_csv("test.csv", header=None).to_numpy(dtype=float)    #load csv
feature_matrix = np.vstack([create_feature_row(row) for row in test_matrix])    #extract features
yhat = model.predict(feature_matrix).astype(int).reshape(-1,1)  #predict
np.savetxt(RESULTS_PATH, yhat, fmt="%d", delimiter=",") #export results