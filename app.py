import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import statsmodels.api as sm

app = Flask(__name__)

# Load the trained model and the scaler
try:
    with open('regmodel.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    model = None
    scaler = None
    print("WARNING: 'regmodel.pkl' or 'scaler.pkl' not found.")
    print("Please ensure you have trained the model and saved the pickle files.")


# These values are needed to inverse-transform the predicted price.
# They are derived from the original training dataset's 'price' column.
MIN_PRICE = 1750000.0
MAX_PRICE = 13300000.0

# The specific order of numerical features the scaler was trained on.
# Note: 'bedrooms' is included here as the scaler expects it, even if the final model doesn't use it.
SCALER_NUM_VARS_ORDER = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'price']

# The final feature columns the regression model (lr_4) expects, in the correct order.
# Note: 'bedrooms' and 'basement' are NOT included here.
MODEL_FINAL_COLUMNS = [
    'const', 'area', 'bathrooms', 'stories', 'mainroad', 'guestroom',
    'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'unfurnished'
]

def make_prediction(input_data):
    """
    Takes a dictionary of features, processes them, and returns the predicted price.
    This function encapsulates the full prediction logic.
    """
    if not model or not scaler:
        raise RuntimeError("Model or scaler not loaded. Cannot make predictions.")

    # --- 1. Process input data ---
    numerical_features = {
        'area': float(input_data['area']),
        'bathrooms': int(input_data['bathrooms']),
        'stories': int(input_data['stories']),
        'parking': int(input_data['parking'])
    }
    
    binary_features = {
        'mainroad': int(input_data['mainroad']),
        'guestroom': int(input_data['guestroom']),
        'hotwaterheating': int(input_data['hotwaterheating']),
        'airconditioning': int(input_data['airconditioning']),
        'prefarea': int(input_data['prefarea'])
    }
    
    furnishing_status = input_data.get('furnishingstatus', 'unfurnished')
    dummy_features = {
        'unfurnished': 1 if furnishing_status == 'unfurnished' else 0
    }

    # --- 2. Scale the numerical features ---
    # Create a temporary DataFrame that matches the structure the scaler was trained on.
    temp_df_for_scaling = pd.DataFrame(columns=SCALER_NUM_VARS_ORDER)
    temp_df_for_scaling.loc[0] = 0 # Initialize a row with zeros
    
    # Populate with user's numerical input and a placeholder for 'bedrooms'
    temp_df_for_scaling.update(pd.DataFrame([numerical_features]))
    temp_df_for_scaling['bedrooms'] = 0 # Dummy value as scaler expects it, but model doesn't.
    
    # Scale the data
    scaled_numerical_array = scaler.transform(temp_df_for_scaling)
    
    # --- 3. Construct the final DataFrame for the model ---
    # Combine all features required by the final model into a dictionary
    all_features_for_model = {
        'area': scaled_numerical_array[0, SCALER_NUM_VARS_ORDER.index('area')],
        'bathrooms': scaled_numerical_array[0, SCALER_NUM_VARS_ORDER.index('bathrooms')],
        'stories': scaled_numerical_array[0, SCALER_NUM_VARS_ORDER.index('stories')],
        'parking': scaled_numerical_array[0, SCALER_NUM_VARS_ORDER.index('parking')],
        **binary_features,
        **dummy_features
    }
    
    # Create DataFrame and add the constant
    final_df = pd.DataFrame([all_features_for_model])
    final_df_with_const = sm.add_constant(final_df, has_constant='add')
    
    # Ensure the column order matches the model's exact expectation
    final_df_ordered = final_df_with_const[MODEL_FINAL_COLUMNS]

    # --- 4. Make prediction and inverse transform ---
    scaled_prediction = model.predict(final_df_ordered)[0]
    
    # Inverse transform the prediction to get the actual price
    predicted_price = scaled_prediction * (MAX_PRICE - MIN_PRICE) + MIN_PRICE
    
    return predicted_price

@app.route('/')
def home():
    """Renders the home page with the input form."""
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction from the HTML form."""
    try:
        form_values = request.form.to_dict()
        predicted_price = make_prediction(form_values)
        prediction_text = f"Predicted House Price: ₹{predicted_price:,.2f}"
    except Exception as e:
        prediction_text = f"An error occurred: {e}"

    return render_template('home.html', prediction_text=prediction_text)

@app.route('/predict_api', methods=['POST'])
def predict_api():
    """Handles prediction from an API call (e.g., Postman)."""
    try:
        api_data = request.json['data']
        predicted_price = make_prediction(api_data)
        return jsonify({'predicted_price': f"₹{predicted_price:,.2f}"})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
