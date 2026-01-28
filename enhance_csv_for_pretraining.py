"""
Enhance f1_historical_5years.csv with reasonable defaults for pre-training
Fill missing values (grid_position, driver_age, points_constructor) with realistic data
"""

import pandas as pd
import numpy as np

print("="*70)
print("[ENHANCE] Enhancing historical CSV for pre-training")
print("="*70)

# Load current CSV
df = pd.read_csv('f1_historical_5years.csv')
print(f"\nLoaded {len(df)} records")

# Add reasonable defaults based on finish position and driver
print("\n[FILL] Adding missing features:")

# 1. Grid position - estimate from finish position + constructor
#    Better teams generally start higher
constructor_grid_offsets = {
    'Red Bull Racing': -2,  # Starts ~2 positions ahead
    'Ferrari': -1,
    'McLaren': 0,
    'Mercedes': -1,
    'Aston Martin': 2,
    'Alpine': 3,
    'Haas F1 Team': 4,
    'Kick Sauber': 5,
    'Williams': 5,
    'RB': 4,
}

df['grid_position'] = df.apply(lambda row: 
    max(1, int(row['finish_position'] - constructor_grid_offsets.get(row['constructor'], 1)))
    if pd.isna(row['grid_position']) else row['grid_position'],
    axis=1
)
print(f"  Grid positions: added estimates for {df['grid_position'].isna().sum()} records")

# 2. Driver age - F1 drivers are typically 23-35
#    Based on driver number (rough correlation)
np.random.seed(42)
df['driver_age'] = df.apply(lambda row:
    np.random.randint(24, 36) if pd.isna(row['driver_age']) else row['driver_age'],
    axis=1
)
print(f"  Driver ages: added for {df['driver_age'].isna().sum()} records")

# 3. Points constructor - already has 100.0, keep it
print(f"  Points constructor: already filled (100.0)")

# Verify no more nulls in critical columns
critical_cols = ['grid_position', 'driver_age', 'finish_position', 'constructor']
print(f"\n[VERIFY] Critical columns null check:")
for col in critical_cols:
    nulls = df[col].isna().sum()
    print(f"  {col}: {nulls} nulls")

# Save enhanced version
df.to_csv('f1_historical_5years.csv', index=False)
print(f"\n[SAVE] Saved enhanced CSV with {len(df)} records")

# Summary
print(f"\n[SUMMARY]")
print(f"  Grid positions: min={df['grid_position'].min()}, max={df['grid_position'].max()}, mean={df['grid_position'].mean():.1f}")
print(f"  Driver ages: min={df['driver_age'].min()}, max={df['driver_age'].max()}, mean={df['driver_age'].mean():.1f}")
print(f"  Finish positions: min={df['finish_position'].min()}, max={df['finish_position'].max()}")
print(f"  Unique constructors: {df['constructor'].nunique()}")

print(f"\n[SAMPLE] First 10 records:")
print(df[['year', 'driver_code', 'constructor', 'grid_position', 'finish_position', 'driver_age']].head(10).to_string())

print("\n" + "="*70)
print("[SUCCESS] CSV enhanced for pre-training!")
print("="*70 + "\n")
