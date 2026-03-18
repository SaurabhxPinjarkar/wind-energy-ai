"""
Step 4: Flask Web Application
Interactive map-based website for wind farm deployment prediction
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import requests
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow for ANN
try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

app = Flask(__name__)

# ==========================================
# LOAD MODELS AND CONFIGURATION
# ==========================================

print("[LOADING RESOURCES]")

# Load configuration
with open("prediction_config.pkl", "rb") as f:
    config = pickle.load(f)

best_model_name = config['best_model']
feature_columns = config['feature_columns']
threshold_energy = config['energy_threshold']

print(f"✓ Best Model: {best_model_name}")
print(f"✓ Energy Threshold: {threshold_energy:.2f}")

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
print("✓ Scaler loaded")

# Load best model
if best_model_name == 'Linear Regression':
    with open("model_lr.pkl", "rb") as f:
        model = pickle.load(f)
elif best_model_name == 'Random Forest':
    with open("model_rf.pkl", "rb") as f:
        model = pickle.load(f)
elif best_model_name == 'XGBoost':
    with open("model_xgb.pkl", "rb") as f:
        model = pickle.load(f)
elif best_model_name == 'ANN':
    if TENSORFLOW_AVAILABLE:
        model = load_model("model_ann.h5")
    else:
        print("   ⚠ ANN selected but TensorFlow not available. Using XGBoost instead.")
        with open("model_xgb.pkl", "rb") as f:
            model = pickle.load(f)
        best_model_name = 'XGBoost'
else:
    with open("model_xgb.pkl", "rb") as f:
        model = pickle.load(f)
    best_model_name = 'XGBoost'

print(f"✓ Model loaded: {best_model_name}")

# Load location data for initial map display
import pandas as pd
from datetime import datetime, timedelta
locations_df = pd.read_csv("maharashtra_100_locations_named.csv")
print(f"✓ Locations loaded: {len(locations_df)} sites")

# ==========================================
# PREDICTION FUNCTIONS
# ==========================================

def fetch_real_time_weather(latitude, longitude):
    """Fetch weather data from NASA POWER API"""
    try:
        # NASA's database is typically 5-7 days trailing real-time for stable meteorological telemetry
        # We fetch the last 10 days of available data and pick the most recent valid day
        end_date = datetime.now() - timedelta(days=5)
        start_date = end_date - timedelta(days=10)
        
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        
        url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=WS10M,T2M,PS,RH2M&community=RE&longitude={longitude}&latitude={latitude}&start={start_str}&end={end_str}&format=JSON"
        response = requests.get(url, timeout=10).json()

        if 'properties' in response:
            data = response['properties']['parameter']

            ws10m_dict = data['WS10M']
            t2m_dict = data['T2M']
            ps_dict = data['PS']
            rh2m_dict = data['RH2M']

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


def predict_energy(latitude, longitude, fetch_real_time=True, ws10m=None, t2m=None, ps=None, rh2m=None):
    """Predict wind energy for a location"""
    
    try:
        # Fetch or use provided weather data
        if fetch_real_time:
            weather = fetch_real_time_weather(latitude, longitude)
            if weather['status'] != 'success':
                return {
                    'status': 'error',
                    'message': f"Failed to fetch weather data"
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
            energy_pred = model.predict(features_scaled, verbose=0)[0][0]
        else:
            energy_pred = model.predict(features_scaled)[0]
        
        # Ensure positive energy value
        energy_pred = max(0, energy_pred)
        
        # Determine suitability
        if energy_pred >= threshold_energy:
            suitability = "✓ Good Location"
            color = "green"
        else:
            suitability = "✗ Not Suitable"
            color = "red"
        
        suitability_score = min(100, (energy_pred / threshold_energy) * 100)
        
        return {
            'status': 'success',
            'latitude': latitude,
            'longitude': longitude,
            'wind_speed': round(ws10m, 2),
            'temperature': round(t2m, 2),
            'pressure': round(ps, 2),
            'humidity': round(rh2m, 2),
            'predicted_energy': round(energy_pred, 2),
            'energy_threshold': round(threshold_energy, 2),
            'suitability': suitability,
            'suitability_score': round(suitability_score, 2),
            'color': color,
            'model_used': best_model_name
        }
    
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Prediction error: {str(e)}"
        }


# ==========================================
# ROUTES
# ==========================================

@app.route('/')
def index():
    """Serve the main webpage"""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.json
        latitude = float(data.get('latitude'))
        longitude = float(data.get('longitude'))
        
        # Fetch real-time data
        result = predict_energy(latitude, longitude, fetch_real_time=True)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


@app.route('/api/locations', methods=['GET'])
def api_locations():
    """Get all 100 pre-selected locations"""
    try:
        locations = []
        for idx, row in locations_df.iterrows():
            locations.append({
                'place': row['Place'],
                'latitude': float(row['Latitude']),
                'longitude': float(row['Longitude'])
            })
        return jsonify({
            'status': 'success',
            'count': len(locations),
            'locations': locations
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/model-info', methods=['GET'])
def api_model_info():
    """Get model information"""
    try:
        with open("best_model_info.pkl", "rb") as f:
            model_info = pickle.load(f)
        
        return jsonify({
            'status': 'success',
            'model': best_model_name,
            'r2_score': round(model_info['test_r2'], 4),
            'rmse': round(model_info['test_rmse'], 4),
            'energy_threshold': round(threshold_energy, 2)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/batch-predict', methods=['POST'])
def api_batch_predict():
    """Batch prediction for multiple locations"""
    try:
        data = request.json
        locations_input = data.get('locations', [])
        
        results = []
        for loc in locations_input:
            lat = float(loc.get('latitude'))
            lon = float(loc.get('longitude'))
            
            result = predict_energy(lat, lon, fetch_real_time=False, 
                                   ws10m=loc.get('ws10m', 4.0),
                                   t2m=loc.get('t2m', 25.0),
                                   ps=loc.get('ps', 101.0),
                                   rh2m=loc.get('rh2m', 70.0))
            results.append(result)
        
        return jsonify({
            'status': 'success',
            'predictions': results
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


if __name__ == '__main__':
    import os
    print("\n" + "=" * 80)
    print("WIND FARM DEPLOYMENT PREDICTION SYSTEM")
    print("=" * 80)
    print(f"\n✓ Best Model: {best_model_name}")
    print(f"✓ 100 Locations Loaded")
    print(f"✓ Flask Server Starting...")
    
    port = int(os.environ.get('PORT', 10000))
    print(f"\n🌐 Server will run on port: {port}")
    print("\n" + "=" * 80)
    
    app.run(host='0.0.0.0', port=port)


