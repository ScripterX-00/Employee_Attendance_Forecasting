## Employee_Attendance_Forecasting
Project Overview

This project focuses on analyzing and forecasting employee attendance behavior, specifically Late Arrival and Early Departure, using historical attendance data.
Machine Learning models are applied to identify patterns, perform predictions, and support workforce planning and HR decision-making.

- Algorithm Used: RandomForest
- Platform: Google Colab


## Dataset Description

The dataset contains approximately 10,000 attendance records

## Technologies Used

- Programming Language: Python
- Libraries:
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Environment: Google Colab

##  Project Structure

```
Employee_Attendance_Forecasting/
│
├── Employee_Attendance_Forecasting.ipynb
├── README.md  # Project documentation
└── dataset.csv   
```
---


## Workflow

- Import Required Libraries
- Load Dataset
- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Feature scaling (important for KNN)
- Train-test split
- Model Training
- Model Evaluation
- Result Analysis

##  Model Details
- Algorithm: Random Forest Classifier
- Output Type: Multi-output (predicts more than one target at a time)

## Implementation

```bash
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
import numpy as np

# === Step 1: Prepare lag features for all employees ===
all_late = []
all_early = []

for empid in df['EmpId'].unique():
    emp_data = df[df['EmpId'] == empid].copy()
    late = create_lag_features(emp_data.copy(), "LateArrival", lags=7).dropna()
    early = create_lag_features(emp_data.copy(), "EarlyDeparture", lags=7).dropna()

    common_dates = set(late['Date']).intersection(set(early['Date']))
    late = late[late['Date'].isin(common_dates)].reset_index(drop=True)
    early = early[early['Date'].isin(common_dates)].reset_index(drop=True)

    for i in range(1, 8):
        late[f'LateArrival_lag{i}'] = late[f'LateArrival_lag{i}'].astype(int)
        early[f'EarlyDeparture_lag{i}'] = early[f'EarlyDeparture_lag{i}'].astype(int)

    if not late.empty and not early.empty:
        features = pd.concat([
            late[[f'LateArrival_lag{i}' for i in range(1, 8)]],
            early[[f'EarlyDeparture_lag{i}' for i in range(1, 8)]]
        ], axis=1)
        targets = pd.DataFrame({
            'LateArrival': late['LateArrival'].values,
            'EarlyDeparture': early['EarlyDeparture'].values
        })
        all_late.append(features)
        all_early.append(targets)

# === Step 2: Train MultiOutput Model ===
X_all = pd.concat(all_late, ignore_index=True)
y_all = pd.concat(all_early, ignore_index=True)

base_rf = RandomForestClassifier(
    max_depth = 12,
    min_samples_leaf=2,
    n_estimators=300,
    random_state=0,
    class_weight='balanced',
    n_jobs=-1
)
multi_model = MultiOutputClassifier(base_rf)
multi_model.fit(X_all, y_all)

# === Step 3: Evaluation ===
y_pred_all = multi_model.predict(X_all)
print("\n=== Evaluation for All Employees ===")
print(f"Exact Match Accuracy: {accuracy_score(y_all, y_pred_all):.2f}")
print(f"Macro F1 Score: {f1_score(y_all, y_pred_all, average='macro', zero_division=0):.2f}")
print("\nClassification Report (Per Label):")
print(classification_report(y_all, y_pred_all, target_names=["LateArrival", "EarlyDeparture"], zero_division=0))



=== Evaluation for All Employees ===
Exact Match Accuracy: 0.75
Macro F1 Score: 0.85

Classification Report (Per Label):
                precision    recall  f1-score   support

   LateArrival       0.89      0.86      0.88      4826
EarlyDeparture       0.83      0.84      0.83      3589

     micro avg       0.86      0.85      0.86      8415
     macro avg       0.86      0.85      0.85      8415
  weighted avg       0.86      0.85      0.86      8415
   samples avg       0.66      0.65      0.65      8415

```
# Prediction Targets

The model simultaneously predicts:
- Late Arrival
- Early Departure
This is achieved using MultiOutputClassifier, which allows one model to handle multiple dependent outputs.

## Model Performance
Overall Accuracy: 75%

## Future Enhancements

- Deep Learning (LSTM / GRU) for time-series forecasting
- Dashboard integration (Power BI / Streamlit)
- Real-time attendance prediction API
- Cloud deployment

##  Author
This project was developed by an Intern Team at Capsitech as part of an internship program during the period 20 July 2025 to 10 August 2025.
I was an active member of the intern development team during the internship period from 1 July 2025 to 15 August 2025.