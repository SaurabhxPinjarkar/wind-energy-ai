"""
Step 1: Data Preprocessing
Load, clean, and prepare the wind dataset for ML training
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("STEP 1: DATA PREPROCESSING")
print("=" * 80)

# Load the dataset
print("\n[1] Loading dataset...")
df = pd.read_csv("final_wind_dataset.csv")
print(f"   Dataset shape: {df.shape}")
print(f"   Columns: {df.columns.tolist()}")

# Display basic info
print("\n[2] Dataset Info:")
print(f"   Missing values:\n{df.isnull().sum()}")
print(f"\n   Data types:\n{df.dtypes}")

# Handle missing values
print("\n[3] Handling missing values...")
initial_rows = len(df)
df = df.dropna()
removed_rows = initial_rows - len(df)
print(f"   Removed {removed_rows} rows with missing values")
print(f"   Remaining rows: {len(df)}")

# Create target variable: Energy (proportional to WS10M^3)
print("\n[4] Creating target variable (Energy)...")
print("   Formula: Energy = (WS10M) ** 3")
df['Energy'] = df['WS10M'] ** 3
print(f"   Energy statistics:")
print(f"   - Min: {df['Energy'].min():.2f}")
print(f"   - Max: {df['Energy'].max():.2f}")
print(f"   - Mean: {df['Energy'].mean():.2f}")
print(f"   - Std: {df['Energy'].std():.2f}")

# Define features and target
print("\n[5] Defining features and target...")
feature_columns = ['WS10M', 'T2M', 'PS', 'RH2M', 'Latitude', 'Longitude']
X = df[feature_columns].copy()
y = df['Energy'].copy()

print(f"   Features (X): {feature_columns}")
print(f"   Target (y): Energy")
print(f"   X shape: {X.shape}")
print(f"   y shape: {y.shape}")

# Standardize features
print("\n[6] Standardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"   Scaler fitted on {len(X_scaled)} samples")

# Create processed dataset
print("\n[7] Creating processed dataset...")
processed_df = pd.DataFrame(X_scaled, columns=feature_columns)
processed_df['Energy'] = y.values
processed_df['Place'] = df['Place'].values

# Save processed data
print("\n[8] Saving processed data...")
processed_df.to_csv("processed_wind_data.csv", index=False)
print("   ✓ Saved: processed_wind_data.csv")

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("   ✓ Saved: scaler.pkl")

# Save feature columns
with open("feature_columns.pkl", "wb") as f:
    pickle.dump(feature_columns, f)
print("   ✓ Saved: feature_columns.pkl")

print("\n" + "=" * 80)
print("DATA PREPROCESSING COMPLETED ✅")
print("=" * 80)
print(f"\nSummary:")
print(f"  - Total records: {len(processed_df)}")
print(f"  - Features: {len(feature_columns)}")
print(f"  - Energy range: {processed_df['Energy'].min():.2f} to {processed_df['Energy'].max():.2f}")
print(f"\nNext: Run 2_model_training.py")
