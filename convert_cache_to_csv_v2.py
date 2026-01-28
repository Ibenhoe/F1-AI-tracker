"""
Convert .cache folder (2020-2024) to f1_historical_5years.csv
Extracts race results from all cached sessions
"""

import os
import pandas as pd
from pathlib import Path
import json

print("="*70)
print("[CONVERTER] FastF1 .cache folder to Historical CSV (2020-2024)")
print("="*70)

CACHE_DIR = '.cache'
all_records = []

if not os.path.exists(CACHE_DIR):
    print(f"ERROR: {CACHE_DIR} folder not found!")
    exit(1)

print(f"\nScanning {CACHE_DIR}...\n")

# Scan all years
for year_dir in sorted(os.listdir(CACHE_DIR)):
    year_path = os.path.join(CACHE_DIR, year_dir)
    
    if not os.path.isdir(year_path):
        continue
    
    try:
        year = int(year_dir)
    except:
        continue
    
    print(f"[YEAR {year}]")
    
    # Scan races in year
    races = [r for r in os.listdir(year_path) if os.path.isdir(os.path.join(year_path, r))]
    
    for race_dir in sorted(races):
        race_path = os.path.join(year_path, race_dir)
        
        # Scan sessions
        sessions = [s for s in os.listdir(race_path) if os.path.isdir(os.path.join(race_path, s))]
        
        for session_dir in sessions:
            session_path = os.path.join(race_path, session_dir)
            
            # Try to load results.json if it exists
            results_file = os.path.join(session_path, 'results.json')
            
            if os.path.exists(results_file):
                try:
                    print(f"  {session_dir:50s}", end=" ... ")
                    
                    with open(results_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Check if it's race results
                    if not isinstance(data, dict) or 'results' not in data:
                        print("[SKIP] Not race results")
                        continue
                    
                    results = data.get('results', [])
                    event_name = data.get('event_name', 'Unknown')
                    round_num = data.get('round', None)
                    
                    driver_count = 0
                    for result in results:
                        try:
                            driver_code = result.get('driver_code', 'UNK')
                            driver_number = result.get('driver_number')
                            finish_pos = result.get('position')
                            grid_pos = result.get('grid_position')
                            constructor = result.get('constructor', 'Unknown')
                            
                            if not finish_pos or finish_pos > 20:
                                continue
                            
                            record = {
                                'year': year,
                                'round': round_num,
                                'event': event_name,
                                'circuit': result.get('circuit', 'Unknown'),
                                'driver_code': str(driver_code).upper(),
                                'driver_number': driver_number,
                                'driver_age': result.get('driver_age'),
                                'constructor': str(constructor),
                                'grid_position': grid_pos,
                                'finish_position': int(finish_pos),
                                'points_constructor': 100.0,
                            }
                            
                            all_records.append(record)
                            driver_count += 1
                        except:
                            pass
                    
                    print(f"[OK] {driver_count} drivers")
                    
                except Exception as e:
                    print(f"[ERROR] {str(e)[:40]}")

print(f"\n[EXTRACT] Total records: {len(all_records)}\n")

if all_records:
    # Create DataFrame
    df = pd.DataFrame(all_records)
    
    # Remove duplicates
    print("[CLEAN] Removing duplicates...")
    initial = len(df)
    df = df.drop_duplicates(subset=['year', 'round', 'driver_code'], keep='first')
    print(f"  Removed {initial - len(df)} duplicates -> {len(df)} records")
    
    # Sort
    df = df.sort_values(['year', 'round']).reset_index(drop=True)
    
    # Null analysis
    print(f"\n[NULL ANALYSIS]")
    for col in df.columns:
        nulls = df[col].isna().sum()
        pct = (nulls / len(df) * 100)
        print(f"  {col:20s}: {nulls:4d} nulls ({pct:5.1f}%)")
    
    # Fill missing grid_positions with estimates (based on finish + constructor)
    print(f"\n[FILL] Filling missing values...")
    
    constructor_grid_bonus = {
        'Red Bull': -2, 'Ferrari': -1, 'Mercedes': -1, 'McLaren': 0,
        'Aston Martin': 1, 'Alpine': 2, 'Haas': 3, 'Williams': 4,
        'Alfa Romeo': 4, 'AlphaTauri': 3, 'Sauber': 3
    }
    
    for idx in df[df['grid_position'].isna()].index:
        finish = df.loc[idx, 'finish_position']
        constructor = df.loc[idx, 'constructor']
        
        # Find best matching constructor key
        bonus = 0
        for key, val in constructor_grid_bonus.items():
            if key.lower() in constructor.lower():
                bonus = val
                break
        
        grid_est = max(1, int(finish - bonus))
        df.loc[idx, 'grid_position'] = grid_est
    
    print(f"  Grid positions filled: {df['grid_position'].isna().sum()} remaining")
    
    # Fill driver ages with random realistic values
    import numpy as np
    np.random.seed(42)
    missing_ages = df[df['driver_age'].isna()].index
    df.loc[missing_ages, 'driver_age'] = np.random.randint(24, 37, len(missing_ages))
    print(f"  Driver ages filled: {df['driver_age'].isna().sum()} remaining")
    
    # Save
    output_file = 'f1_historical_5years.csv'
    print(f"\n[SAVE] Writing {output_file}...")
    df.to_csv(output_file, index=False)
    
    # Summary
    print(f"\n[SUMMARY]")
    print(f"  File: {output_file}")
    print(f"  Shape: {df.shape}")
    print(f"  Years: {sorted(df['year'].unique())}")
    print(f"  Races: {len(df.groupby(['year', 'round']))}")
    print(f"  Drivers (unique): {len(df.groupby('driver_code'))}")
    print(f"  Avg drivers/race: {len(df) / max(1, len(df.groupby(['year', 'round']))):.1f}")
    
    print(f"\n[DATA QUALITY]")
    print(f"  Grid positions: min={df['grid_position'].min()}, max={df['grid_position'].max()}")
    print(f"  Finish positions: min={df['finish_position'].min()}, max={df['finish_position'].max()}")
    print(f"  Driver ages: min={df['driver_age'].min()}, max={df['driver_age'].max()}, mean={df['driver_age'].mean():.1f}")
    
    print(f"\n[SAMPLE] Sample records from each year:")
    for year in sorted(df['year'].unique()):
        sample = df[df['year'] == year].head(2)
        print(f"\n  {year}:")
        for _, row in sample.iterrows():
            print(f"    {row['event']:15s} - {row['driver_code']} ({row['constructor']:15s}) Pos {row['finish_position']}")
    
    print("\n" + "="*70)
    print("[SUCCESS] Historical CSV created from .cache!")
    print("="*70 + "\n")
else:
    print("[ERROR] No data extracted!")
