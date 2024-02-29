import streamlit as st

import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Create a simple Streamlit UI
st.title('Random Forest Regression App')

# Load the trained machine learning model
with open('rf_model', 'rb') as f:
    model = pickle.load(f)

# Function to predict resale price based on user inputs
def predict_resale_price(lease_commence_date,floor_area_sqm):
    # Preprocess user inputs if necessary
    # For simplicity, let's assume the inputs are already preprocessed
    # Concatenate the user inputs into a feature vector
    features = [lease_commence_date,floor_area_sqm]
    # Reshape the feature vector for prediction
    features = [features]  # Convert to 2D array
    # Use the trained model to make predictions
    predicted_price = model.predict(features)
    return predicted_price

# Streamlit web application
def main():
    # Set title and description
    st.title('Singapore HDB Resale Price Prediction')
    st.write('This app predicts the resale price of HDB flats in Singapore based on user inputs.')

    # Input form for user details
    st.sidebar.header('Input Details')
    floor_area_sqm = st.sidebar.number_input('Floor Area (sqm)', min_value=1, max_value=300, value=50)
    lease_commence_date = st.sidebar.number_input('lease_commence_date', min_value=1966, max_value=2012, value=2000)
    
    # Predict button
    if st.sidebar.button('Predict'):
        # Call predict_resale_price function with user inputs
        predicted_price = predict_resale_price(lease_commence_date,floor_area_sqm)
        # Display prediction
        st.subheader('Predicted Resale Price')
        st.write('Estimated Price: $', round(predicted_price[0], 2))

if __name__ == '__main__':
    main()