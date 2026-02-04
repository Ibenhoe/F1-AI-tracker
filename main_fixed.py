#!/usr/bin/env python3
"""
FIXED MAIN.PY - With Debug Output
All feature engineering is properly executed BEFORE training
"""

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import traceback

print("\n" + "="*80)
print("🚀 MAIN.PY - FIXED VERSION WITH DEBUG OUTPUT")
print("="*80 + "\n")

# ---------------------------------------------------------
# STAP 1: DATA LADEN & FEATURE ENGINEERING
# ---------------------------------------------------------

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(script_dir, "unprocessed_f1_training_data.csv"))
    print(f"✓ Data loaded: {len(df)} rows")
except FileNotFoundError:
    print("❌ FOUT: Run eerst 'data_script.py' om de data te genereren!")
    exit()

# Filter 2015+
df = df[df['year'] >= 2015].copy()
print(f"✓ Filtered to 2015+: {len(df)} rows\n")

print("="*80)
print("FEATURE ENGINEERING PROGRESS")
print("="*80 + "\n")

# === FEATURE 1: DRIVER AGE ===
print("[1/25] Creating: driver_age")
try:
    df['dob'] = pd.to_datetime(df['dob'])
    df['date'] = pd.to_datetime(df['date'])
    df['driver_age'] = (df['date'] - df['dob']).dt.days / 365.25
    print("     ✓ driver_age created\n")
except Exception as e:
    print(f"     ❌ ERROR: {e}\n")
    traceback.print_exc()

# === FEATURE 2: DRIVER EXPERIENCE ===
print("[2/25] Creating: driver_experience")
try:
    df = df.sort_values(by='date')
    df['driver_experience'] = df.groupby('driverId').cumcount()
    print("     ✓ driver_experience created\n")
except Exception as e:
    print(f"     ❌ ERROR: {e}\n")

# === FEATURE 3: HOME RACE ===
print("[3/25] Creating: is_home_race")
try:
    if 'country' not in df.columns:
        print("     ⚠️  'country' column missing, trying to load from circuits.csv...")
        circuits = pd.read_csv('../F1_data_mangement/circuits.csv')
        df = pd.merge(df, circuits[['circuitId', 'country']], on='circuitId', how='left')
        print("     ✓ Merged country from circuits.csv")
    
    nationality_map = {
        'British': 'UK', 'German': 'Germany', 'Spanish': 'Spain', 'French': 'France',
        'Italian': 'Italy', 'Dutch': 'Netherlands', 'Australian': 'Australia',
        'Monegasque': 'Monaco', 'American': 'USA', 'Japanese': 'Japan', 'Canadian': 'Canada',
        'Mexican': 'Mexico', 'Brazilian': 'Brazil'
    }
    df['mapped_nationality'] = df['nationality'].map(nationality_map).fillna(df['nationality'])
    df['is_home_race'] = np.where(df['mapped_nationality'] == df['country'], 1, 0)
    print("     ✓ is_home_race created\n")
except Exception as e:
    print(f"     ⚠️  WARNING (continuing): {e}\n")
    df['is_home_race'] = 0

# === FEATURE 4-5: BINNING ===
print("[4/25] Creating: grid_bin_code & age_bin_code")
try:
    le_grid = LabelEncoder()
    le_age = LabelEncoder()
    
    df['grid_bin'] = pd.cut(df['grid'], bins=[-1, 1, 3, 10, 15, 25], labels=['Pole', 'Top3', 'Points', 'Midfield', 'Back'])
    df['age_bin'] = pd.cut(df['driver_age'], bins=[17, 24, 30, 36, 60], labels=['Rookie', 'Prime', 'Experienced', 'Veteran'])
    
    df['grid_bin_code'] = le_grid.fit_transform(df['grid_bin'].astype(str))
    df['age_bin_code'] = le_age.fit_transform(df['age_bin'].astype(str))
    print("     ✓ grid_bin_code & age_bin_code created\n")
except Exception as e:
    print(f"     ❌ ERROR: {e}\n")
    traceback.print_exc()

# === FEATURE 6: TEAM CONTINUITY ===
print("[6/25] Creating: team_continuity_id & related features")
try:
    def get_team_continuity_id(row):
        name = str(row.get('name_team', '')).lower()
        cid = row['constructorId']
        
        if 'mercedes' in name or 'brawn' in name or 'honda' in name or 'bar' in name: return 131
        if 'red bull' in name or 'jaguar' in name or 'stewart' in name: return 9
        if 'alpine' in name or 'renault' in name or 'lotus' in name: return 214
        if 'aston martin' in name or 'racing point' in name or 'force india' in name or 'spyker' in name or 'jordan' in name: return 117
        if 'rb' in name or 'alphatauri' in name or 'toro rosso' in name or 'minardi' in name: return 213
        if 'sauber' in name or 'alfa romeo' in name: return 15
        
        return cid
    
    df['team_continuity_id'] = df.apply(get_team_continuity_id, axis=1)
    print("     ✓ team_continuity_id created\n")
except Exception as e:
    print(f"     ⚠️  WARNING (using constructorId): {e}\n")
    df['team_continuity_id'] = df['constructorId']

# === FEATURE 7: OVERTAKING DIFFICULTY ===
print("[7/25] Creating: circuit_overtake_index")
try:
    df['pos_change_abs'] = (df['grid'] - df['positionOrder']).abs()
    df['circuit_overtake_index'] = df.groupby('circuitId')['pos_change_abs'].transform('mean')
    df['circuit_overtake_index'] = df['circuit_overtake_index'].fillna(df['pos_change_abs'].mean())
    print("     ✓ circuit_overtake_index created\n")
except Exception as e:
    print(f"     ❌ ERROR: {e}\n")

# === FEATURE 8-9: RECENT FORM (CRITICAL) ===
print("[8/25] Creating: driver_recent_form & constructor_recent_form")
try:
    def calculate_rolling_avg(group, window=5):
        def weighted_mean(x):
            weights = np.arange(1, len(x) + 1)
            return np.average(x, weights=weights)
        return group.shift(1).rolling(window, min_periods=1).apply(weighted_mean, raw=True)
    
    df['driver_recent_form'] = df.groupby('driverId')['positionOrder'].transform(calculate_rolling_avg)
    df['constructor_recent_form'] = df.groupby('team_continuity_id')['positionOrder'].transform(calculate_rolling_avg)
    
    df['driver_recent_form'] = df['driver_recent_form'].fillna(12.0)
    df['constructor_recent_form'] = df['constructor_recent_form'].fillna(12.0)
    print("     ✓ driver_recent_form & constructor_recent_form created\n")
except Exception as e:
    print(f"     ❌ ERROR: {e}\n")
    df['driver_recent_form'] = 12.0
    df['constructor_recent_form'] = 12.0

# === FEATURE 10-11: PIT STOPS ===
print("[10/25] Creating: driver_recent_pit_avg & constructor_recent_pit_avg")
try:
    if 'pit_stops_duration_ms' in df.columns and 'pit_stops_count' in df.columns:
        df['pit_stops_count'] = df['pit_stops_count'].fillna(0)
        df['pit_stops_duration_ms'] = df['pit_stops_duration_ms'].fillna(0)
        
        df['avg_pit_duration'] = df.apply(lambda x: x['pit_stops_duration_ms'] / x['pit_stops_count'] if x['pit_stops_count'] > 0 else np.nan, axis=1)
        
        global_pit_mean = df['avg_pit_duration'].mean()
        if pd.isna(global_pit_mean): global_pit_mean = 24000.0
        
        df['avg_pit_duration_filled'] = df['avg_pit_duration'].fillna(global_pit_mean)
        
        df['driver_recent_pit_avg'] = df.groupby('driverId')['avg_pit_duration_filled'].transform(calculate_rolling_avg)
        df['constructor_recent_pit_avg'] = df.groupby('team_continuity_id')['avg_pit_duration_filled'].transform(calculate_rolling_avg)
        
        df['driver_recent_pit_avg'] = df['driver_recent_pit_avg'].fillna(global_pit_mean)
        df['constructor_recent_pit_avg'] = df['constructor_recent_pit_avg'].fillna(global_pit_mean)
        print("     ✓ pit stop features created\n")
    else:
        print("     ⚠️  No pit stop data, using defaults\n")
        global_pit_mean = 24000.0
        df['driver_recent_pit_avg'] = global_pit_mean
        df['constructor_recent_pit_avg'] = global_pit_mean
except Exception as e:
    print(f"     ⚠️  WARNING (using defaults): {e}\n")
    df['driver_recent_pit_avg'] = 24000.0
    df['constructor_recent_pit_avg'] = 24000.0

# === FEATURE 12-13: RELIABILITY (DNF) ===
print("[12/25] Creating: driver_dnf_rate & constructor_dnf_rate")
try:
    if 'status' in df.columns:
        df['is_dnf'] = ~df['status'].astype(str).str.match(r'(Finished|\+\d+\sLaps)').fillna(False)
        df['is_dnf'] = df['is_dnf'].astype(int)
        
        df['driver_dnf_rate'] = df.groupby('driverId')['is_dnf'].transform(calculate_rolling_avg)
        df['constructor_dnf_rate'] = df.groupby('team_continuity_id')['is_dnf'].transform(calculate_rolling_avg)
        
        df['driver_dnf_rate'] = df['driver_dnf_rate'].fillna(0.2)
        df['constructor_dnf_rate'] = df['constructor_dnf_rate'].fillna(0.2)
        print("     ✓ DNF features created\n")
    else:
        print("     ⚠️  No status data, using defaults\n")
        df['driver_dnf_rate'] = 0.2
        df['constructor_dnf_rate'] = 0.2
except Exception as e:
    print(f"     ⚠️  WARNING (using defaults): {e}\n")
    df['driver_dnf_rate'] = 0.2
    df['constructor_dnf_rate'] = 0.2

# === FEATURE 14: TEAMMATE COMPARISON ===
print("[14/25] Creating: driver_vs_team_form")
try:
    df['driver_vs_team_form'] = df['constructor_recent_form'] - df['driver_recent_form']
    print("     ✓ driver_vs_team_form created\n")
except Exception as e:
    print(f"     ❌ ERROR: {e}\n")
    df['driver_vs_team_form'] = 0.0

# === FEATURE 15-17: WEATHER ===
print("[15/25] Creating: weather features")
try:
    if 'weather_temp_c' in df.columns:
        df['weather_temp_c'] = df['weather_temp_c'].fillna(df['weather_temp_c'].mean())
        df['weather_precip_mm'] = df['weather_precip_mm'].fillna(0.0)
        df['weather_cloud_pct'] = df['weather_cloud_pct'].fillna(50.0)
    else:
        df['weather_temp_c'] = 20.0
        df['weather_precip_mm'] = 0.0
        df['weather_cloud_pct'] = 50.0
    print("     ✓ Weather features created\n")
except Exception as e:
    print(f"     ⚠️  WARNING: {e}\n")

# === FEATURE 18-20: TRACK HISTORY ===
print("[18/25] Creating: driver_track_* features")
try:
    def calculate_expanding_mean(group):
        return group.shift(1).expanding().mean()
    
    def calculate_expanding_sum(group):
        return group.shift(1).expanding().sum()
    
    track_groups = df.groupby(['driverId', 'circuitId'])
    
    df['driver_track_avg_grid'] = track_groups['grid'].transform(calculate_expanding_mean)
    df['driver_track_avg_finish'] = track_groups['positionOrder'].transform(calculate_expanding_mean)
    
    df['is_podium'] = (df['positionOrder'] <= 3).astype(int)
    df['driver_track_podiums'] = track_groups['is_podium'].transform(calculate_expanding_sum)
    
    df['driver_track_avg_grid'] = df['driver_track_avg_grid'].fillna(12.0)
    df['driver_track_avg_finish'] = df['driver_track_avg_finish'].fillna(12.0)
    df['driver_track_podiums'] = df['driver_track_podiums'].fillna(0.0)
    print("     ✓ Track history features created\n")
except Exception as e:
    print(f"     ❌ ERROR: {e}\n")
    df['driver_track_avg_grid'] = 12.0
    df['driver_track_avg_finish'] = 12.0
    df['driver_track_podiums'] = 0.0

# === FEATURE 21-22: SPEED RATIO & CIRCUIT SPEED ===
print("[21/25] Creating: driver_recent_speed & circuit_speed_index")
try:
    def parse_fastest_lap(t_str):
        if pd.isna(t_str) or str(t_str).strip() == '\\N': return np.nan
        try:
            parts = str(t_str).split(':')
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            return float(t_str)
        except:
            return np.nan
    
    if 'fastestLapTime' in df.columns:
        df['fastest_lap_seconds'] = df['fastestLapTime'].apply(parse_fastest_lap)
        
        race_best_times = df.groupby('raceId')['fastest_lap_seconds'].min().reset_index().rename(columns={'fastest_lap_seconds': 'race_best_time'})
        df = pd.merge(df, race_best_times, on='raceId', how='left')
        
        df['speed_ratio'] = df['fastest_lap_seconds'] / df['race_best_time']
        df['speed_ratio'] = df['speed_ratio'].fillna(1.10)
        
        df['driver_recent_speed'] = df.groupby('driverId')['speed_ratio'].transform(calculate_rolling_avg)
        df['driver_recent_speed'] = df['driver_recent_speed'].fillna(1.10)
        
        df['circuit_speed_index'] = df.groupby('circuitId')['fastest_lap_seconds'].transform('mean')
        global_lap_mean = df['fastest_lap_seconds'].mean()
        df['circuit_speed_index'] = df['circuit_speed_index'].fillna(global_lap_mean)
        print("     ✓ Speed features created\n")
    else:
        df['driver_recent_speed'] = 1.10
        df['circuit_speed_index'] = 90.0
        print("     ⚠️  No lap time data, using defaults\n")
except Exception as e:
    print(f"     ⚠️  WARNING: {e}\n")
    df['driver_recent_speed'] = 1.10
    df['circuit_speed_index'] = 90.0

# === FEATURE 23: POINTS BEFORE RACE ===
print("[23/25] Creating: points_before_race & grid_penalty")
try:
    df['points_before_race'] = (df.groupby(['year', 'driverId'])['points'].cumsum() - df['points']).fillna(0)
    
    if 'quali_position' in df.columns:
        df['quali_pos_filled'] = df['quali_position'].fillna(df['grid'])
    else:
        df['quali_pos_filled'] = df['grid']
    
    df['grid_penalty'] = df['grid'] - df['quali_pos_filled']
    print("     ✓ points_before_race & grid_penalty created\n")
except Exception as e:
    print(f"     ❌ ERROR: {e}\n")
    df['points_before_race'] = 0.0
    df['grid_penalty'] = 0.0

print("\n" + "="*80)
print("✅ FEATURE ENGINEERING COMPLETE!")
print("="*80 + "\n")

# --- CHECK WHICH FEATURES EXIST ---
feature_cols = [
    'grid', 'grid_penalty', 'circuitId', 'circuit_speed_index',
    'circuit_overtake_index', 'driver_age', 'driver_experience',
    'is_home_race', 'grid_bin_code', 'age_bin_code',
    'driver_recent_form', 'constructor_recent_form', 'driver_vs_team_form',
    'driver_track_avg_grid', 'driver_track_avg_finish', 'driver_track_podiums',
    'weather_temp_c', 'weather_precip_mm', 'weather_cloud_pct',
    'driver_recent_pit_avg', 'constructor_recent_pit_avg',
    'driver_dnf_rate', 'constructor_dnf_rate',
    'driver_recent_speed', 'points_before_race'
]

available = [f for f in feature_cols if f in df.columns]
missing = [f for f in feature_cols if f not in df.columns]

print(f"✓ AVAILABLE: {len(available)}/{len(feature_cols)} features")
print(f"✗ MISSING:   {len(missing)}/{len(feature_cols)} features\n")

if missing:
    print("Missing features:")
    for f in missing:
        print(f"   - {f}")
    print()

# ---------------------------------------------------------
# STAP 2: TRAINING DATA PREPARATION
# ---------------------------------------------------------

print("="*80)
print("TRAINING DATA PREPARATION")
print("="*80 + "\n")

# Use available features only
feature_cols = [f for f in feature_cols if f in df.columns]

# Select training data
train_df = df.dropna(subset=['positionOrder'] + feature_cols)

print(f"✓ Training samples: {len(train_df)}/{len(df)} ({(len(train_df)/len(df))*100:.1f}%)")
print(f"✓ Features: {len(feature_cols)}")
print(f"✓ Target variable: positionOrder\n")

X = train_df[feature_cols]
y = train_df['positionOrder']

print("X shape:", X.shape)
print("Y shape:", y.shape)
print("\nData ready for training!")

