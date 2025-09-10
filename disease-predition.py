# -*- coding: utf-8 -*-
"""Disease Prediction Model (Local Version with File Picker)"""

import pandas as pd
from tkinter import Tk, filedialog
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ----------------------------------------------------------------
# Open file picker to select CSV
# ----------------------------------------------------------------
Tk().withdraw()  # Hide main tkinter window
file_path = filedialog.askopenfilename(
    title="Select the Disease Symptoms CSV file",
    filetypes=[("CSV files", "*.csv")]
)

if not file_path:
    raise FileNotFoundError("No file selected!")

# Load dataset
df = pd.read_csv(file_path)

print("First 5 rows of dataset:")
print(df.head())
print("\nColumns in dataset:")
print(df.columns)

# ----------------------------------------------------------------
# Preprocess the Data (encode categorical features)
# ----------------------------------------------------------------
label_encoders = {}
categorical_columns = ['Disease', 'Fever', 'Cough', 'Fatigue', 'Difficulty Breathing',
                       'Gender', 'Blood Pressure', 'Cholesterol Level', 'Outcome Variable']

for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

print("\nFirst 5 rows after encoding:")
print(df.head())

# ----------------------------------------------------------------
# Split the data into features (X) and target (y)
# ----------------------------------------------------------------
X = df.drop('Outcome Variable', axis=1)
y = df['Outcome Variable']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------------------------------
# Train a Classification Model
# ----------------------------------------------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ----------------------------------------------------------------
# Evaluate the Model
# ----------------------------------------------------------------
y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision : {precision_score(y_test, y_pred):.2f}")
print(f"Recall    : {recall_score(y_test, y_pred):.2f}")
print(f"F1 Score  : {f1_score(y_test, y_pred):.2f}")

# ----------------------------------------------------------------
# Example Predictions with New Data
# ----------------------------------------------------------------
def make_prediction(new_data: dict):
    """Encode new data, predict, and return decoded label"""
    new_data_encoded = {}
    for column, value in new_data.items():
        if column in label_encoders:
            new_data_encoded[column] = label_encoders[column].transform([value])[0]
        else:
            new_data_encoded[column] = value

    new_data_df = pd.DataFrame([new_data_encoded], columns=X.columns)
    prediction = model.predict(new_data_df)
    prediction_decoded = label_encoders['Outcome Variable'].inverse_transform(prediction)
    return prediction_decoded[0]

# Example usage
sample1 = {
    'Disease': 'Influenza',
    'Fever': 'Yes',
    'Cough': 'No',
    'Fatigue': 'Yes',
    'Difficulty Breathing': 'Yes',
    'Age': 20,
    'Gender': 'Female',
    'Blood Pressure': 'Low',
    'Cholesterol Level': 'Normal'
}
print(f"\nPrediction for sample1: {make_prediction(sample1)}")