"""
Step 2: Model Training
Train multiple ML models and compare performance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow for ANN
try:
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("\n[WARNING] TensorFlow not available - ANN model will be skipped")

print("=" * 80)
print("STEP 2: MODEL TRAINING")
print("=" * 80)

# Load processed data
print("\n[1] Loading processed data...")
df = pd.read_csv("processed_wind_data.csv")
X = df.drop(['Energy', 'Place'], axis=1).values
y = df['Energy'].values
print(f"   X shape: {X.shape}")
print(f"   y shape: {y.shape}")

# Train-test split (80-20)
print("\n[2] Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

# Dictionary to store models and results
models = {}
results = {}

# ==========================================
# MODEL 1: LINEAR REGRESSION
# ==========================================
print("\n" + "=" * 80)
print("MODEL 1: LINEAR REGRESSION")
print("=" * 80)
print("\n[Training]")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
print("   ✓ Training completed")

# Predictions
y_train_pred_lr = lr_model.predict(X_train)
y_test_pred_lr = lr_model.predict(X_test)

# Evaluation
train_rmse_lr = np.sqrt(mean_squared_error(y_train, y_train_pred_lr))
test_rmse_lr = np.sqrt(mean_squared_error(y_test, y_test_pred_lr))
train_r2_lr = r2_score(y_train, y_train_pred_lr)
test_r2_lr = r2_score(y_test, y_test_pred_lr)

print("\n[Evaluation]")
print(f"   Train RMSE: {train_rmse_lr:.4f}")
print(f"   Test RMSE:  {test_rmse_lr:.4f}")
print(f"   Train R²:   {train_r2_lr:.4f}")
print(f"   Test R²:    {test_r2_lr:.4f}")

models['Linear Regression'] = lr_model
results['Linear Regression'] = {
    'train_rmse': train_rmse_lr,
    'test_rmse': test_rmse_lr,
    'train_r2': train_r2_lr,
    'test_r2': test_r2_lr
}

# Save model
with open("model_lr.pkl", "wb") as f:
    pickle.dump(lr_model, f)
print("\n   ✓ Model saved: model_lr.pkl")

# ==========================================
# MODEL 2: RANDOM FOREST
# ==========================================
print("\n" + "=" * 80)
print("MODEL 2: RANDOM FOREST REGRESSOR")
print("=" * 80)
print("\n[Training]")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=0
)
rf_model.fit(X_train, y_train)
print("   ✓ Training completed")

# Predictions
y_train_pred_rf = rf_model.predict(X_train)
y_test_pred_rf = rf_model.predict(X_test)

# Evaluation
train_rmse_rf = np.sqrt(mean_squared_error(y_train, y_train_pred_rf))
test_rmse_rf = np.sqrt(mean_squared_error(y_test, y_test_pred_rf))
train_r2_rf = r2_score(y_train, y_train_pred_rf)
test_r2_rf = r2_score(y_test, y_test_pred_rf)

print("\n[Evaluation]")
print(f"   Train RMSE: {train_rmse_rf:.4f}")
print(f"   Test RMSE:  {test_rmse_rf:.4f}")
print(f"   Train R²:   {train_r2_rf:.4f}")
print(f"   Test R²:    {test_r2_rf:.4f}")

# Feature importance
print("\n[Feature Importance]")
feature_names = ['WS10M', 'T2M', 'PS', 'RH2M', 'Latitude', 'Longitude']
for name, importance in zip(feature_names, rf_model.feature_importances_):
    print(f"   {name}: {importance:.4f}")

models['Random Forest'] = rf_model
results['Random Forest'] = {
    'train_rmse': train_rmse_rf,
    'test_rmse': test_rmse_rf,
    'train_r2': train_r2_rf,
    'test_r2': test_r2_rf
}

# Save model
with open("model_rf.pkl", "wb") as f:
    pickle.dump(rf_model, f)
print("\n   ✓ Model saved: model_rf.pkl")

# ==========================================
# MODEL 3: XGBOOST
# ==========================================
print("\n" + "=" * 80)
print("MODEL 3: XGBOOST REGRESSOR")
print("=" * 80)
print("\n[Training]")
xgb_model = XGBRegressor(
    n_estimators=100,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)
xgb_model.fit(X_train, y_train)
print("   ✓ Training completed")

# Predictions
y_train_pred_xgb = xgb_model.predict(X_train)
y_test_pred_xgb = xgb_model.predict(X_test)

# Evaluation
train_rmse_xgb = np.sqrt(mean_squared_error(y_train, y_train_pred_xgb))
test_rmse_xgb = np.sqrt(mean_squared_error(y_test, y_test_pred_xgb))
train_r2_xgb = r2_score(y_train, y_train_pred_xgb)
test_r2_xgb = r2_score(y_test, y_test_pred_xgb)

print("\n[Evaluation]")
print(f"   Train RMSE: {train_rmse_xgb:.4f}")
print(f"   Test RMSE:  {test_rmse_xgb:.4f}")
print(f"   Train R²:   {train_r2_xgb:.4f}")
print(f"   Test R²:    {test_r2_xgb:.4f}")

# Feature importance
print("\n[Feature Importance]")
for name, importance in zip(feature_names, xgb_model.feature_importances_):
    print(f"   {name}: {importance:.4f}")

models['XGBoost'] = xgb_model
results['XGBoost'] = {
    'train_rmse': train_rmse_xgb,
    'test_rmse': test_rmse_xgb,
    'train_r2': train_r2_xgb,
    'test_r2': test_r2_xgb
}

# Save model
with open("model_xgb.pkl", "wb") as f:
    pickle.dump(xgb_model, f)
print("\n   ✓ Model saved: model_xgb.pkl")

# ==========================================
# MODEL 4: ARTIFICIAL NEURAL NETWORK (ANN)
# ==========================================
if TENSORFLOW_AVAILABLE:
    print("\n" + "=" * 80)
    print("MODEL 4: ARTIFICIAL NEURAL NETWORK (ANN)")
    print("=" * 80)
    print("\n[Building Architecture]")
    ann_model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.1),
        Dense(1, activation='linear')
    ])
    print("   ✓ Model architecture built")
    print("   Layers: Input(6) → Dense(64, relu) → Dropout(0.2) → Dense(32, relu)")
    print("           → Dropout(0.2) → Dense(16, relu) → Dropout(0.1) → Output(1)")

    print("\n[Compiling]")
    ann_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    print("   ✓ Model compiled (Optimizer: Adam, Loss: MSE)")

    print("\n[Training]")
    history = ann_model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    print("   ✓ Training completed (100 epochs)")

    # Predictions
    y_train_pred_ann = ann_model.predict(X_train, verbose=0).flatten()
    y_test_pred_ann = ann_model.predict(X_test, verbose=0).flatten()

    # Evaluation
    train_rmse_ann = np.sqrt(mean_squared_error(y_train, y_train_pred_ann))
    test_rmse_ann = np.sqrt(mean_squared_error(y_test, y_test_pred_ann))
    train_r2_ann = r2_score(y_train, y_train_pred_ann)
    test_r2_ann = r2_score(y_test, y_test_pred_ann)

    print("\n[Evaluation]")
    print(f"   Train RMSE: {train_rmse_ann:.4f}")
    print(f"   Test RMSE:  {test_rmse_ann:.4f}")
    print(f"   Train R²:   {train_r2_ann:.4f}")
    print(f"   Test R²:    {test_r2_ann:.4f}")

    models['ANN'] = ann_model
    results['ANN'] = {
        'train_rmse': train_rmse_ann,
        'test_rmse': test_rmse_ann,
        'train_r2': train_r2_ann,
        'test_r2': test_r2_ann
    }

    # Save model
    ann_model.save("model_ann.h5")
    print("\n   ✓ Model saved: model_ann.h5")
else:
    print("\n" + "=" * 80)
    print("MODEL 4: ARTIFICIAL NEURAL NETWORK (ANN) - SKIPPED")
    print("=" * 80)
    print("\nTensorFlow is not installed. To include ANN model, install tensorflow:")
    print("   pip install tensorflow")
    print("\nContinuing with 3 models: Linear Regression, Random Forest, XGBoost")

# ==========================================
# MODEL COMPARISON
# ==========================================
print("\n" + "=" * 80)
print("MODEL COMPARISON & SELECTION")
print("=" * 80)

comparison_df = pd.DataFrame(results).T
print("\n[Performance Metrics]")
print(comparison_df.round(4))

# Find best model by test RMSE
best_model_name = comparison_df['test_rmse'].idxmin()
best_rmse = comparison_df.loc[best_model_name, 'test_rmse']
best_r2 = comparison_df.loc[best_model_name, 'test_r2']

print("\n" + "=" * 80)
print(f"🏆 BEST MODEL: {best_model_name}")
print("=" * 80)
print(f"   Test RMSE: {best_rmse:.4f}")
print(f"   Test R²:   {best_r2:.4f}")

# Save comparison results
comparison_df.to_csv("model_comparison.csv")
print(f"\n   ✓ Comparison saved: model_comparison.csv")

# Save results summary
with open("best_model_info.pkl", "wb") as f:
    pickle.dump({
        'best_model': best_model_name,
        'test_rmse': best_rmse,
        'test_r2': best_r2,
        'all_results': results
    }, f)
print("   ✓ Best model info saved: best_model_info.pkl")

print("\n" + "=" * 80)
print("MODEL TRAINING COMPLETED ✅")
print("=" * 80)
print(f"\nNext: Run 3_model_deployment.py")
