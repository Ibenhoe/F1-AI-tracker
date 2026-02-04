#!/usr/bin/env python3
"""
DIAGNOSTIC SCRIPT: Diagnose why changes in main.py don't show effect

This script identifies:
1. Which features main.py ACTUALLY uses for training
2. Which features are being ENGINEERED but then DROPPED
3. Data quality issues that prevent features from being used
4. Cache/model loading issues
"""

import pandas as pd
import numpy as np
import os

print("\n" + "="*70)
print("🔍 DIAGNOSTIC: Checking main.py feature engineering")
print("="*70 + "\n")

# Load data (same as main.py does)
script_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(script_dir, "unprocessed_f1_training_data.csv"))
print(f"✓ Data loaded: {len(df)} rows")

# Filter (same as main.py)
df = df[df['year'] >= 2015].copy()
print(f"✓ Filtered to 2015+: {len(df)} rows\n")

# Check 1: Basic columns
print("="*70)
print("CHECK 1: BASIC COLUMNS (Grid, Position, Age, etc)")
print("="*70)
required_basic = ['grid', 'positionOrder', 'date', 'dob', 'driverId', 'circuitId', 'constructorId', 'year']
for col in required_basic:
    exists = col in df.columns
    missing = df[col].isna().sum() if exists else "N/A"
    print(f"  {col:20s} | Exists: {str(exists):5s} | Missing: {missing if isinstance(missing, str) else f'{missing:5d}'}")

# Check 2: Feature engineering columns
print("\n" + "="*70)
print("CHECK 2: ENGINEERED FEATURES (Latest additions)")
print("="*70)

# Create features (same logic as main.py STAP 1)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date')
df['driver_experience'] = df.groupby('driverId').cumcount()

# Recent form
def calculate_rolling_avg(group, window=5):
    def weighted_mean(x):
        weights = np.arange(1, len(x) + 1)
        return np.average(x, weights=weights)
    return group.shift(1).rolling(window, min_periods=1).apply(weighted_mean, raw=True)

df['driver_recent_form'] = df.groupby('driverId')['positionOrder'].transform(calculate_rolling_avg)
df['driver_recent_form'] = df['driver_recent_form'].fillna(12.0)

# Check these features
engineered_features = ['driver_experience', 'driver_recent_form']
for col in engineered_features:
    exists = col in df.columns
    missing = df[col].isna().sum()
    sample_values = df[col].head(3).tolist()
    print(f"  {col:25s} | Exists: {str(exists):5s} | Missing: {missing:5d} | Sample: {sample_values}")

# Check 3: The BIG FEATURES (new pit stops, DNF, speed ratio, etc)
print("\n" + "="*70)
print("CHECK 3: ADVANCED FEATURES (pit stops, DNF, speed ratio)")
print("="*70)

advanced_features = ['pit_stops_duration_ms', 'pit_stops_count', 'status', 'fastestLapTime', 'quali_position']
for col in advanced_features:
    exists = col in df.columns
    missing = df[col].isna().sum() if exists else "N/A"
    print(f"  {col:30s} | Exists: {str(exists):5s} | Missing: {missing if isinstance(missing, str) else f'{missing:5d}'}")

# Check 4: feature_cols from main.py (THE ACTUAL FEATURES USED FOR TRAINING)
print("\n" + "="*70)
print("CHECK 4: FEATURES DEFINED IN main.py TRAINING (line ~270)")
print("="*70)

feature_cols = [
    'grid', 
    'grid_penalty',
    'circuitId', 
    'circuit_speed_index',
    'circuit_overtake_index',
    'driver_age',
    'driver_experience',
    'is_home_race',
    'grid_bin_code',
    'age_bin_code',
    'driver_recent_form',
    'constructor_recent_form',
    'driver_vs_team_form',
    'driver_track_avg_grid',
    'driver_track_avg_finish',
    'driver_track_podiums',
    'weather_temp_c',
    'weather_precip_mm',
    'weather_cloud_pct',
    'driver_recent_pit_avg',
    'constructor_recent_pit_avg',
    'driver_dnf_rate',
    'constructor_dnf_rate',
    'driver_recent_speed',
    'points_before_race'
]

print(f"\n👉 Total features in feature_cols: {len(feature_cols)}\n")
missing_features = [f for f in feature_cols if f not in df.columns]
available_features = [f for f in feature_cols if f in df.columns]

print(f"✓ AVAILABLE in DataFrame:     {len(available_features):2d} features")
print(f"✗ MISSING from DataFrame:     {len(missing_features):2d} features")

if missing_features:
    print(f"\n⚠️  THESE FEATURES ARE MISSING (won't be used in training):")
    for f in missing_features:
        print(f"     - {f}")

# Check 5: What happens when we try to select these features?
print("\n" + "="*70)
print("CHECK 5: TRAINING DATA AVAILABILITY")
print("="*70)

# Try to select features (same as main.py does)
try:
    # First, only select AVAILABLE features
    available_feature_cols = [f for f in feature_cols if f in df.columns]
    print(f"Using {len(available_feature_cols)}/{len(feature_cols)} available features")
    
    # Try to drop rows with NaN in required columns
    train_df_attempt = df.dropna(subset=['positionOrder'] + available_feature_cols)
    print(f"After dropna on features: {len(train_df_attempt)} rows available for training")
    print(f"Data loss: {len(df) - len(train_df_attempt)} rows ({(1 - len(train_df_attempt)/len(df))*100:.1f}%)")
    
    if len(train_df_attempt) == 0:
        print("❌ ERROR: NO TRAINING DATA AVAILABLE!")
        print("\n   Possible reasons:")
        print("   1. Too many missing values in features")
        print("   2. Features are not being created properly")
        print("   3. Data filtering removed all usable rows")
        
except Exception as e:
    print(f"❌ ERROR when selecting features: {e}")

# Check 6: Data quality per feature
print("\n" + "="*70)
print("CHECK 6: DATA QUALITY (Missing values %)")
print("="*70)

critical_features = ['grid', 'positionOrder', 'driver_experience', 'driver_recent_form', 
                     'circuitId', 'driver_age']

for col in critical_features:
    if col in df.columns:
        missing_pct = (df[col].isna().sum() / len(df)) * 100
        print(f"  {col:25s} | Missing: {missing_pct:5.1f}%")
    else:
        print(f"  {col:25s} | ❌ NOT IN DATAFRAME")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("""
If you see ✗ MISSING features above, that's why your changes don't work:
- You engineered the feature, but it's not in the DataFrame when training happens
- The feature is created AFTER the training section, or
- The feature engineering code is throwing an exception silently

SOLUTIONS:
1. Check if feature creation code runs BEFORE train_df = df.dropna(...)
2. Check if there are try/except blocks silently catching errors
3. Add debugging print() statements to see which features are actually created
""")
