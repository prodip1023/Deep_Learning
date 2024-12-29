import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle


# Load the trained model
try:
    model = tf.keras.models.load_model('regression_model.h5')
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()
## load the encoder and scaler
try:
    with open('onehot_encoder_geo_reg.pkl','rb') as file:
        onehot_encoder_geo=pickle.load(file)

    with open('label_encoder_gender_reg.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)

    with open('scaler_reg.pkl', 'rb') as file:
        scaler = pickle.load(file)
except Exception as e:
    st.error(f"Error loading the encoders and scalers: {e}")
    st.stop()

st.title('Customer Salary Prediction')


# User Input fields 
geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
age = st.slider('🎂 Age', 18, 92, value=30)  # Default age set to 30
balance = st.number_input('💰 Balance', min_value=0.0, value=0.0, step=100.0)
credit_score = st.number_input('📊 Credit Score', min_value=300, max_value=850, value=650)
# estimated_salary = st.number_input('💵 Estimated Salary', min_value=0.0, value=50000.0, step=1000.0)
exited = st.selectbox('💳 Has exited',[0,1])
tenure = st.slider('📅 Tenure (Years)', 0, 10, value=5)
num_of_products = st.slider('📦 Number of Products', 1, 4, value=5)
has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1])
is_active_member = st.selectbox('✅ Is Active Member', [0, 1])


# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})
# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
try:
    input_data_scaled = scaler.transform(input_data)
except Exception as e:
    st.error(f"Error scaling the input data: {e}")
    st.stop()


# Predict the Salary
try:
    predicted_salary = model.predict(input_data_scaled)[0][0]
    st.success(f"💵 Predicted Salary: {predicted_salary:,.2f}")
except Exception as e:
    st.error(f"Error during prediction: {e}")
