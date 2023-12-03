import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import joblib  # Import joblib for model persistence

# Load BMI dataset
bmi_dataset = pd.read_csv('bmi.csv')  # Replace 'your_file.csv' with the actual file path

# Encoding 'BmiClass' to numerical values
label_encoder = LabelEncoder()
bmi_dataset['BmiClassEncoded'] = label_encoder.fit_transform(bmi_dataset['BmiClass'])

# Separating the data and labels
X_bmi = bmi_dataset[['Age', 'Bmi']]
Y_height = bmi_dataset['Height']
Y_weight = bmi_dataset['Weight']

# Standardize the data
scaler_bmi = StandardScaler()
standardized_data_bmi = scaler_bmi.fit_transform(X_bmi)
X_bmi = standardized_data_bmi

# Train-test split for Height prediction
X_train_height, X_test_height, Y_train_height, Y_test_height = train_test_split(X_bmi, Y_height, test_size=0.2, random_state=2)

# RandomForestRegressor for Height prediction
height_model = RandomForestRegressor(n_estimators=100, random_state=2)
height_model.fit(X_train_height, Y_train_height)

# Train-test split for Weight prediction
X_train_weight, X_test_weight, Y_train_weight, Y_test_weight = train_test_split(X_bmi, Y_weight, test_size=0.2, random_state=2)

# RandomForestRegressor for Weight prediction
weight_model = RandomForestRegressor(n_estimators=100, random_state=2)
weight_model.fit(X_train_weight, Y_train_weight)

# Save the models using joblib
joblib.dump(height_model, 'height_model.joblib')
joblib.dump(weight_model, 'weight_model.joblib')

# Function to determine BMI class based on ranges
def determine_bmi_class(bmi_value):
    if bmi_value < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi_value <= 24.9:
        return 'Healthy Weight'
    elif 25 <= bmi_value <= 29.9:
        return 'Overweight'
    else:
        return 'Obese'

# Streamlit App
st.title("BMI Prediction App")

# Input form for user
st.sidebar.header("Enter BMI and Age:")
bmi_input = st.sidebar.slider("BMI", 10, 40, 20)  # Ask only for BMI
age_input = st.sidebar.slider("Age", 18, 100, 30)

# Load the pre-trained models
loaded_height_model = joblib.load('height_model.joblib')
loaded_weight_model = joblib.load('weight_model.joblib')

# Predicting Height and Weight based on entered BMI and Age
input_data_bmi = np.array([[age_input, bmi_input]])
std_data_bmi_input = scaler_bmi.transform(input_data_bmi)

# Predict Height
predicted_height = loaded_height_model.predict(std_data_bmi_input)

# Predict Weight
predicted_weight = loaded_weight_model.predict(std_data_bmi_input)

# Display the results
st.header("BMI Prediction:")
st.write(f'Entered BMI: {bmi_input} - {determine_bmi_class(bmi_input)}')
st.write(f'Entered Age: {age_input} years')
st.write(f'Predicted Height: {predicted_height[0]:.2f} m')
st.write(f'Predicted Weight: {predicted_weight[0]:.2f} kg')
