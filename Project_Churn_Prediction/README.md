# Customer Churn Prediction using ANN
This repository contains a Streamlit application that predicts customer churn using a trained Artificial Neural Network (ANN) model. The app takes user input for various features, processes it using pre-trained encoders and a scaler, and outputs the churn probability.

# Features
- **Streamlit App:** User-friendly interface for entering customer details.
- **Model Integration:** Predicts churn probability using a trained ANN model (model.h5).
- **Preprocessing Pipelines:** Utilizes pre-trained encoders and a scaler for input transformation.
- **Real-Time Predictions:** Displays churn probability based on user inputs.

# Prerequisites
Before running the app, ensure you have the following installed:

- Python 3.7 or higher
- Required Python libraries (see requirements.txt)

# Files in the Repository
- **model.h5:** Trained ANN model saved using TensorFlow/Keras.
- **label_encoder_gender.pkl:** Label encoder for the Gender feature.
- **onehot_encoder_geo.pkl:** One-hot encoder for the Geography feature.
- **scaler.pkl:** Scaler for normalizing input data.
- **app.py:** Streamlit app script for customer churn prediction.
- **requirements.txt:** List of required Python libraries.

# User Inputs in the App
The app collects the following inputs:

- **Geography:** Dropdown for selecting the region.
- **Gender:** Dropdown for selecting gender.
- **Age:** Slider to set the age (18–92).
- **Balance:** Numeric input for account balance.
- **Credit Score:** Numeric input for credit score.
- **Estimated Salary:** Numeric input for salary estimate.
- **Tenure:** Slider to set tenure (0–10 years).
- **Number of Products:** Slider to select the number of products (1–4).
- **Has Credit Card:** Dropdown to indicate if the customer has a credit card.
- **Is Active Member:** Dropdown to indicate if the customer is an active member.

# Outputs
- **Churn Probability:** Displays the likelihood of the customer churning.
- **Prediction:** Indicates whether the customer is likely to churn or not.

# Model Training
The ANN model was trained on customer data using TensorFlow/Keras. It includes:

- Input features preprocessed using encoders and a scaler.
- A multi-layer neural network with activation functions tuned for binary classification.
