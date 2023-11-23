import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st

# Load BMI dataset
bmi_dataset = pd.read_csv('bmi.csv')  # Replace 'your_file.csv' with the actual file path

# Encoding 'BmiClass' to numerical values
label_encoder = LabelEncoder()
bmi_dataset['BmiClassEncoded'] = label_encoder.fit_transform(bmi_dataset['BmiClass'])

# Separating the data and labels
X_bmi = bmi_dataset[['Age', 'Height', 'Weight']]
Y_bmi = bmi_dataset['Bmi']

# Standardize the data
scaler_bmi = StandardScaler()
standardized_data_bmi = scaler_bmi.fit_transform(X_bmi)
X_bmi = standardized_data_bmi

# Train-test split
X_train_bmi, X_test_bmi, Y_train_bmi, Y_test_bmi = train_test_split(X_bmi, Y_bmi, test_size=0.2, random_state=2)

# Linear Regression Model for BMI prediction
bmi_model = LinearRegression()
bmi_model.fit(X_train_bmi, Y_train_bmi)

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
st.sidebar.header("Enter User Information:")
age = st.sidebar.slider("Age", 18, 100, 30)
height = st.sidebar.slider("Height (m)", 1.4, 2.0, 1.7)
weight = st.sidebar.slider("Weight (kg)", 40, 150, 70)

# User input data
input_data_bmi = np.array([[age, height, weight]])
std_data_bmi = scaler_bmi.transform(input_data_bmi)

# BMI Prediction
predicted_bmi = bmi_model.predict(std_data_bmi)
predicted_bmi = round(predicted_bmi[0], 2)

# Calculate Actual BMI
actual_bmi = weight / (height ** 2)
actual_bmi = round(actual_bmi, 2)

# Determine BMI class for actual and predicted BMI
actual_bmi_class = determine_bmi_class(actual_bmi)
predicted_bmi_class = determine_bmi_class(predicted_bmi)

# Display the results
st.header("BMI Prediction:")
st.write(f'Actual BMI: {actual_bmi} - {actual_bmi_class}')
st.write(f'Predicted BMI: {predicted_bmi} - {predicted_bmi_class}')

# Show creativity based on BMI range
if actual_bmi_class == 'Healthy Weight' and predicted_bmi_class == 'Healthy Weight':
    st.success("Congratulations! You are in the Healthy Weight range. Keep up the good work! 😊🎉")
elif actual_bmi_class == 'Underweight' and predicted_bmi_class == 'Underweight':
    st.error("You're in the Underweight range. Make sure to maintain a balanced diet for a healthy lifestyle. 💪🥗")
# Add more conditions and messages for different BMI ranges
elif actual_bmi_class == 'Overweight' and predicted_bmi_class == 'Overweight':
    st.warning("You're in the Overweight range. Consider adopting a healthier lifestyle for overall well-being. 🏋️‍♂️🥦")
elif actual_bmi_class == 'Obese' and predicted_bmi_class == 'Obese':
    st.error("You're in the Obese range. It's important to focus on a healthy lifestyle for overall well-being. 🏃‍♀️🥗")