# cgm-ml-classifier
Supervised ML classifier that detects meal vs. no-meal periods from continuous glucose monitoring data.

# Technologies Used
- **Python**
- **pandas**
- **NumPy**
- **scikit-learn**

# Program Description
This coursework project implements a machine learning pipeline for continuous glucose monitoring (CGM) data. The system processes CGM and insulin pump records to extract meal and no-meal windows, derive discriminative features, and train a classifier that can identify meal events.

## Key Technical Implementation:

- **Data preprocessing**:  
  - Import CGM and Insulin datasets  
  - Generate timestamps and align records  
  - Handle missing values with interpolation and threshold rules  

- **Feature extraction**:  
  - Shape-based features (range, normalized difference, quarter-slope, time-to-peak)  
  - Derivative-based features (max first- and second-order slopes)  
  - Frequency-based feature (FFT power)  

- **Model training**:  
  - Construct feature matrices for meal and no-meal windows  
  - Train an SVM classifier with scaling and class weighting  
  - Evaluate using stratified k-fold cross-validation  

- **Prediction and output**:  
  - Save trained model (`model.pkl`)  
  - `test.py` loads hidden `test.csv` during grading  
  - Generates `Result.csv` with 1 (meal) or 0 (no-meal) per row  
