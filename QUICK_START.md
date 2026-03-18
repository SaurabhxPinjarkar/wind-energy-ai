# QUICK START GUIDE
# Wind Farm Deployment Prediction System

## 🚀 STEP-BY-STEP EXECUTION

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Internet connection (for NASA API)

---

## STEP 1: Install Dependencies

Run this command in PowerShell:

```powershell
pip install -r requirements.txt
```

Expected output:
```
Successfully installed flask-2.3.3 pandas-2.0.3 numpy-1.24.3 ...
```

Time required: 3-5 minutes

---

## STEP 2: Data Preprocessing

Run the preprocessing script:

```powershell
python 1_data_preprocessing.py
```

**What it does**:
- Loads 131,492 records from final_wind_dataset.csv
- Removes missing values
- Creates energy target variable (WS10M^3)
- Standardizes all features
- Saves processed data

**Expected output**:
- processed_wind_data.csv (cleaned data)
- scaler.pkl (feature normalizer)
- feature_columns.pkl (feature names)

**Time**: 30-60 seconds

---

## STEP 3: Train ML Models

Run the training script:

```powershell
python 2_model_training.py
```

**What it does**:
- Trains 4 different ML models
- Evaluates each model on test set
- Compares RMSE and R² scores
- Automatically selects best model
- Saves all trained models

**Expected output**:
```
================================ MODEL 1: LINEAR REGRESSION ================================
Train RMSE: 18.5234
Test RMSE:  19.2145
Train R²:   0.6234
Test R²:    0.6012

================================ MODEL 2: RANDOM FOREST REGRESSOR ================================
Train RMSE: 8.4567
Test RMSE:  10.1234
Train R²:   0.8834
Test R²:    0.8523

... (XGBoost and ANN results) ...

================================ 🏆 BEST MODEL: Random Forest ================================
Test RMSE: 10.1234
Test R²:   0.8523
```

**Files created**:
- model_lr.pkl
- model_rf.pkl
- model_xgb.pkl
- model_ann.h5
- best_model_info.pkl
- model_comparison.csv

**Time**: 5-15 minutes (depending on hardware)

---

## STEP 4: Launch Web Application

Run the Flask server:

```powershell
python 4_website.py
```

**Expected output**:
```
================================================================================
WIND FARM DEPLOYMENT PREDICTION SYSTEM
================================================================================

✓ Best Model: Random Forest
✓ 100 Locations Loaded
✓ Flask Server Starting...

🌐 Open your browser and go to:
   http://localhost:5000

================================================================================
```

---

## STEP 5: Use the Interactive Website

1. **Open browser**: Go to `http://localhost:5000`

2. **Click on the map** anywhere to get predictions

3. **View results** in the sidebar:
   - Predicted energy value
   - Weather data (wind, temperature, pressure, humidity)
   - Suitability status (Green = Good, Red = Not Suitable)

4. **Manual prediction**: Enter custom lat/lon coordinates

5. **Show all locations**: View all 100 pre-selected sites

---

## 📊 EXPECTED RESULTS

### Model Performance

Best model will typically achieve:
- **Test RMSE**: 6-12 kWh
- **Test R² Score**: 0.80-0.92
- **Fastest inference**: < 1 second per prediction

### Web Interface

- Interactive map with 100 location markers
- Real-time weather data from NASA API
- Color-coded suitability (Green/Red)
- Responsive sidebar with detailed results
- Works on desktop and tablet

---

## 🔍 VERIFICATION CHECKLIST

After each step, verify:

✓ Step 1 (Preprocessing)
  - [ ] processed_wind_data.csv exists (>100 MB)
  - [ ] scaler.pkl exists
  - [ ] feature_columns.pkl exists

✓ Step 2 (Training)
  - [ ] model_lr.pkl exists
  - [ ] model_rf.pkl exists
  - [ ] model_xgb.pkl exists
  - [ ] model_ann.h5 exists
  - [ ] best_model_info.pkl exists
  - [ ] model_comparison.csv has results for 4 models

✓ Step 3 (Website)
  - [ ] Flask server runs without errors
  - [ ] No port conflicts (port 5000 available)
  - [ ] Website loads at http://localhost:5000

✓ Step 4 (Functionality)
  - [ ] Map displays Maharashtra region
  - [ ] Clicking map shows predictions
  - [ ] Sidebar shows weather data
  - [ ] Color-coded markers appear (green/red)

---

## 🐛 TROUBLESHOOTING

### Issue: "ModuleNotFoundError: No module named 'flask'"
**Solution**: Run `pip install -r requirements.txt`

### Issue: "File not found: final_wind_dataset.csv"
**Solution**: Ensure both CSV files are in the same directory:
- final_wind_dataset.csv
- maharashtra_100_locations_named.csv

### Issue: "Port 5000 already in use"
**Solution**: 
```powershell
# Find and kill process using port 5000
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

### Issue: "NASA API timeout"
**Solution**: Try a different location, check internet connection

### Issue: "TensorFlow not found"
**Solution**: This might take longer to install
```powershell
pip install --upgrade tensorflow
```

---

## 📈 EXPECTED EXECUTION TIMES

| Step | Task | Time |
|------|------|------|
| 1 | Data Preprocessing | 1 min |
| 2 | Model Training | 10-15 min |
| 3 | Server Startup | 5 sec |
| 4 | Web Interface Load | 2-3 sec |
| 5 | Per-Location Prediction | 2-5 sec |

**Total First-Time Setup**: ~15-20 minutes

---

## 🎯 SUCCESS INDICATORS

You'll know it's working when:

✅ Training shows 4 model results with RMSE and R² scores
✅ Website loads at http://localhost:5000
✅ Map displays Maharashtra with 100 location dots
✅ Clicking map shows prediction results in sidebar
✅ Suitability badge appears (Green/Red)
✅ Progress bar shows energy suitability score

---

## 📝 NEXT STEPS

After successful setup:

1. **Explore predictions** for different Maharashtra locations
2. **Test manual coordinates** using the input fields
3. **Check model comparison** (model_comparison.csv)
4. **Review API endpoints** (/api/predict, /api/locations, etc.)
5. **Consider enhancements** for production deployment

---

## 💾 SAVING RESULTS

To export results:

```powershell
# View model comparison
type model_comparison.csv

# View best model info
python -c "import pickle; print(pickle.load(open('best_model_info.pkl', 'rb')))"
```

---

## 🌐 ACCESSING FROM OTHER DEVICES

To access the website from other computers on your network:

1. Find your IP address:
```powershell
ipconfig | findstr "IPv4"
```

2. Access from another device:
```
http://<YOUR_IP>:5000
```

Example: `http://192.168.1.100:5000`

---

## 📞 SUPPORT

If you encounter issues:

1. **Check README.md** for detailed documentation
2. **Review error messages** - they're descriptive
3. **Verify file permissions** in the project directory
4. **Ensure all dependencies** are installed correctly
5. **Check internet connection** for API calls

---

**Last Updated**: March 18, 2026  
**Status**: Ready to Execute ✅
