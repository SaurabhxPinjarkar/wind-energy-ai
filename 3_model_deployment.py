"""
Step 3: Model Deployment & Prediction Engine
Load best model and create prediction functions for API calls
"""

import pandas as pd
import numpy as np
import pickle
import requests
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow for ANN
try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

print("=" * 80)
print("STEP 3: MODEL DEPLOYMENT & PREDICTION ENGINE")
print("=" * 80)

# Load best model info
print("\n[1] Loading best model information...")
with open("best_model_info.pkl", "rb") as f:
    model_info = pickle.load(f)

best_model_name = model_info['best_model']
print(f"   Best Model: {best_model_name}")
print(f"   Test RMSE: {model_info['test_rmse']:.4f}")
print(f"   Test R²:   {model_info['test_r2']:.4f}")

# Load models and scaler
print("\n[2] Loading trained models...")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
print("   ✓ Scaler loaded")

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)
print(f"   ✓ Feature columns loaded: {feature_columns}")

# Load all models
if best_model_name == 'Linear Regression':
    with open("model_lr.pkl", "rb") as f:
        best_model = pickle.load(f)
elif best_model_name == 'Random Forest':
    with open("model_rf.pkl", "rb") as f:
        best_model = pickle.load(f)
elif best_model_name == 'XGBoost':
    with open("model_xgb.pkl", "rb") as f:
        best_model = pickle.load(f)
elif best_model_name == 'ANN':
    if TENSORFLOW_AVAILABLE:
        best_model = load_model("model_ann.h5")
    else:
        print("   ⚠ ANN selected but TensorFlow not available. Using XGBoost instead.")
        with open("model_xgb.pkl", "rb") as f:
            best_model = pickle.load(f)
        best_model_name = 'XGBoost'
else:
    with open("model_xgb.pkl", "rb") as f:
        best_model = pickle.load(f)
    best_model_name = 'XGBoost'

print(f"   ✓ Best model loaded: {best_model_name}")

# Load processed data to determine threshold
print("\n[3] Calculating energy threshold...")
df = pd.read_csv("processed_wind_data.csv")
energy_values = df['Energy'].values
median_energy = np.median(energy_values)
# Setting a more realistic threshold for Maharashtra specifically
# The 50th percentile (median) makes 50% of historical data points "Good"
threshold_energy = np.percentile(energy_values, 50)  # Changed from 70th to 50th percentile

print(f"   Energy Statistics:")
print(f"   - Min: {energy_values.min():.2f}")
print(f"   - Max: {energy_values.max():.2f}")
print(f"   - Mean: {energy_values.mean():.2f}")
print(f"   - Median: {median_energy:.2f}")
print(f"   - 70th percentile (threshold): {threshold_energy:.2f}")

# ==========================================
# PREDICTION ENGINE
# ==========================================

def fetch_real_time_weather(latitude, longitude):
    """
    Fetch real-time weather data from NASA POWER API
    
    Args:
        latitude (float): Location latitude
        longitude (float): Location longitude
    
    Returns:
        dict: Weather data including WS10M, T2M, PS, RH2M
    """
    try:
        url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=WS10M,T2M,PS,RH2M&community=RE&longitude={longitude}&latitude={latitude}&start=20240101&end=20240318&format=JSON"
        response = requests.get(url, timeout=10).json()
        
        if 'properties' in response:
            data = response['properties']['parameter']
            
            # Get most recent available data
            ws10m_dict = data['WS10M']
            t2m_dict = data['T2M']
            ps_dict = data['PS']
            rh2m_dict = data['RH2M']
            
            # Get the last available values
            ws10m = float(list(ws10m_dict.values())[-1]) if ws10m_dict else 0
            t2m = float(list(t2m_dict.values())[-1]) if t2m_dict else 0
            ps = float(list(ps_dict.values())[-1]) if ps_dict else 0
            rh2m = float(list(rh2m_dict.values())[-1]) if rh2m_dict else 0
            
            return {
                'WS10M': ws10m,
                'T2M': t2m,
                'PS': ps,
                'RH2M': rh2m,
                'status': 'success'
            }
        else:
            return {'status': 'error', 'message': 'Invalid API response'}
    
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


def predict_wind_energy(latitude, longitude, fetch_real_time=True, ws10m=None, t2m=None, ps=None, rh2m=None):
    """
    Predict wind energy potential for a location
    
    Args:
        latitude (float): Location latitude
        longitude (float): Location longitude
        fetch_real_time (bool): Whether to fetch real-time data from NASA API
        ws10m, t2m, ps, rh2m (float): Optional manual weather values
    
    Returns:
        dict: Prediction results including energy value and suitability
    """
    
    try:
        # Fetch or use provided weather data
        if fetch_real_time:
            weather = fetch_real_time_weather(latitude, longitude)
            if weather['status'] != 'success':
                return {
                    'status': 'error',
                    'message': f"Failed to fetch weather data: {weather.get('message', 'Unknown error')}"
                }
            ws10m = weather['WS10M']
            t2m = weather['T2M']
            ps = weather['PS']
            rh2m = weather['RH2M']
        
        # Create feature vector
        features = np.array([[ws10m, t2m, ps, rh2m, latitude, longitude]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        if best_model_name == 'ANN' and TENSORFLOW_AVAILABLE:
            energy_pred = best_model.predict(features_scaled, verbose=0)[0][0]
        else:
            energy_pred = best_model.predict(features_scaled)[0]
        
        # Ensure positive energy value
        energy_pred = max(0, energy_pred)
        
        # Determine suitability
        if energy_pred >= threshold_energy:
            suitability = "Good"
            suitability_score = min(100, (energy_pred / threshold_energy) * 100)
        else:
            suitability = "Not Suitable"
            suitability_score = (energy_pred / threshold_energy) * 100
        
        return {
            'status': 'success',
            'latitude': latitude,
            'longitude': longitude,
            'wind_speed': float(ws10m),
            'temperature': float(t2m),
            'pressure': float(ps),
            'humidity': float(rh2m),
            'predicted_energy': float(round(energy_pred, 2)),
            'energy_threshold': float(round(threshold_energy, 2)),
            'suitability': suitability,
            'suitability_score': float(round(suitability_score, 2)),
            'model_used': best_model_name,
            'model_accuracy': f"R² = {model_info['test_r2']:.4f}"
        }
    
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Prediction failed: {str(e)}"
        }


# Test the prediction engine
print("\n[4] Testing prediction engine...")
print("\n[Test 1] Ratnagiri location (16.9, 73.3)")
result1 = predict_wind_energy(16.9, 73.3, fetch_real_time=False, ws10m=4.5, t2m=25.0, ps=101.0, rh2m=70.0)
print(f"   Status: {result1['status']}")
if result1['status'] == 'success':
    print(f"   Predicted Energy: {result1['predicted_energy']}")
    print(f"   Suitability: {result1['suitability']} ({result1['suitability_score']:.1f}%)")

print("\n[Test 2] Pune location (19.0, 75.0)")
result2 = predict_wind_energy(19.0, 75.0, fetch_real_time=False, ws10m=3.8, t2m=28.0, ps=101.5, rh2m=65.0)
print(f"   Status: {result2['status']}")
if result2['status'] == 'success':
    print(f"   Predicted Energy: {result2['predicted_energy']}")
    print(f"   Suitability: {result2['suitability']} ({result2['suitability_score']:.1f}%)")

# Save prediction engine configuration
print("\n[5] Saving prediction configuration...")
config = {
    'best_model': best_model_name,
    'feature_columns': feature_columns,
    'energy_threshold': threshold_energy,
    'model_r2': model_info['test_r2'],
    'model_rmse': model_info['test_rmse']
}

with open("prediction_config.pkl", "wb") as f:
    pickle.dump(config, f)
print("   ✓ Config saved: prediction_config.pkl")

print("\n" + "=" * 80)
print("DEPLOYMENT COMPLETED ✅")
print("=" * 80)
print(f"\nPrediction functions are ready!")
print(f"Next: Run 4_website.py to start the web server")
