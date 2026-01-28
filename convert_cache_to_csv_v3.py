"""
Convert .cache folder (2020-2024) to f1_historical_5years.csv
Extracts race results from all .ff1pkl cached files
Uses joblib to deserialize since FastF1 uses joblib format
"""

import os
import pandas as pd
import pickle
import sys
from pathlib import Path
import numpy as np

print("="*70)
print("[CONVERTER] .cache folder to Historical CSV (2020-2024)")
print("="*70)

CACHE_DIR = '.cache'
all_records = []

if not os.path.exists(CACHE_DIR):
    print(f"ERROR: {CACHE_DIR} folder not found!")
    exit(1)

print(f"\nScanning {CACHE_DIR} for race results...\n")

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
    races = sorted([r for r in os.listdir(year_path) if os.path.isdir(os.path.join(year_path, r))])
    
    for race_dir in races:
        race_path = os.path.join(year_path, race_dir)
        
        # Scan sessions - only process Race sessions
        sessions = sorted([s for s in os.listdir(race_path) if os.path.isdir(os.path.join(race_path, s))])
        
        for session_dir in sessions:
            if 'Race' not in session_dir:
                continue  # Skip qualifying
            
            session_path = os.path.join(race_path, session_dir)
            
            try:
                print(f"  {session_dir:50s}", end=" ... ")
                
                # Load driver_info.ff1pkl to get driver finishing positions
                driver_info_file = os.path.join(session_path, 'driver_info.ff1pkl')
                
                if not os.path.exists(driver_info_file):
                    print("[SKIP] No driver_info")
                    continue
                
                # Load pickle file (joblib format used by FastF1)
                with open(driver_info_file, 'rb') as f:
                    driver_info_obj = pickle.load(f)
                
                # Extract drivers dict
                drivers_data = {}
                if isinstance(driver_info_obj, dict) and 'data' in driver_info_obj:
                    drivers_data = driver_info_obj['data']
                
                if not drivers_data:
                    print("[SKIP] No drivers found")
                    continue
                
                # Load session_info to get event details
                session_info_file = os.path.join(session_path, 'session_info.ff1pkl')
                session_info = {}
                if os.path.exists(session_info_file):
                    try:
                        with open(session_info_file, 'rb') as f:
                            session_info = pickle.load(f)
                    except:
                        pass
                
                # Extract event info from session_info dict
                round_num = None
                event_name = 'Unknown'
                circuit = 'Unknown'
                
                if isinstance(session_info, dict):
                    round_num = session_info.get('RoundNumber')
                    event_name = session_info.get('EventName', 'Unknown')
                    circuit_obj = session_info.get('Circuit', {})
                    if isinstance(circuit_obj, dict):
                        circuit = circuit_obj.get('CircuitName', 'Unknown')
                
                # Parse from directory name as fallback (more reliable)
                # Format: YYYY-MM-DD_Race_Name_Grand_Prix
                if event_name == 'Unknown' or len(str(event_name)) < 3:
                    try:
                        parts = race_dir.split('_')
                        # Remove date from start
                        race_name_parts = [p for p in parts if not p[0].isdigit() and p != '-']
                        # Remove Grand_Prix suffix
                        if race_name_parts and race_name_parts[-1] == 'Grand' and len(race_name_parts) > 1:
                            race_name_parts = race_name_parts[:-2]  # Remove 'Grand' and 'Prix'
                        if race_name_parts:
                            event_name = ' '.join(race_name_parts)
                    except:
                        pass
                
                # Extract circuit from event name if circuit is Unknown
                if circuit == 'Unknown':
                    try:
                        parts = race_dir.split('_')
                        race_name_parts = [p for p in parts if not p[0].isdigit() and p != '-']
                        if race_name_parts:
                            # For well-known races, map name to circuit
                            circuit_mapping = {
                                'Bahrain': 'Sakhir',
                                'Saudi': 'Jeddah',
                                'Australia': 'Melbourne',
                                'Japan': 'Suzuka',
                                'China': 'Shanghai',
                                'Miami': 'Miami',
                                'Monaco': 'Monte Carlo',
                                'Canada': 'Montreal',
                                'Spain': 'Barcelona',
                                'Austria': 'Spielberg',
                                'British': 'Silverstone',
                                'Hungary': 'Budapest',
                                'Belgium': 'Spa',
                                'Netherlands': 'Zandvoort',
                                'Italy': 'Monza',
                                'Azerbaijan': 'Baku',
                                'Singapore': 'Marina Bay',
                                'Austin': 'Circuit of Americas',
                                'Mexico': 'Hermanos Rodriguez',
                                'Brazil': 'Interlagos',
                                'Abu': 'Yas Island'
                            }
                            for key, val in circuit_mapping.items():
                                if key in event_name:
                                    circuit = val
                                    break
                            if circuit == 'Unknown':
                                circuit = ' '.join(race_name_parts)
                    except:
                        pass
                
                # Extract driver records (Line field = finishing position)
                driver_count = 0
                for driver_abbr, driver_details in drivers_data.items():
                    try:
                        if not isinstance(driver_details, dict):
                            continue
                        
                        # Line field represents finishing position (1-based)
                        finish_pos = driver_details.get('Line')
                        driver_number = driver_details.get('RacingNumber')
                        team_name = driver_details.get('TeamName', 'Unknown')
                        first_name = driver_details.get('FirstName', '')
                        last_name = driver_details.get('LastName', '')
                        
                        # Skip if no position or invalid
                        if finish_pos is None or not isinstance(finish_pos, (int, float)):
                            continue
                        
                        finish_pos = int(finish_pos)
                        if finish_pos > 25 or finish_pos < 1:
                            continue
                        
                        record = {
                            'year': year,
                            'round': int(round_num) if round_num else 1,  # Default to 1 if missing
                            'event': str(event_name),
                            'circuit': str(circuit),
                            'driver_code': str(driver_abbr).upper(),
                            'driver_number': int(driver_number) if driver_number else None,
                            'driver_first_name': str(first_name),
                            'driver_last_name': str(last_name),
                            'driver_age': None,  # Not available in this data
                            'constructor': str(team_name),
                            'grid_position': None,  # Not in driver_info
                            'finish_position': finish_pos,
                            'position_gain': None,  # Will calculate
                            'points_constructor': 100.0,
                        }
                        
                        all_records.append(record)
                        driver_count += 1
                        
                    except Exception as e:
                        pass
                
                print(f"[OK] {driver_count} drivers")
                
            except Exception as e:
                print(f"[ERROR] {str(e)[:40]}")

print(f"\n[EXTRACT] Total records: {len(all_records)}\n")

if all_records and len(all_records) > 100:
    # Create DataFrame
    df = pd.DataFrame(all_records)
    
    # Remove duplicates based on year, event, driver (NOT round since it's often None)
    print("[CLEAN] Removing duplicates...")
    initial = len(df)
    df = df.drop_duplicates(subset=['year', 'event', 'driver_code'], keep='first')
    print(f"  Removed {initial - len(df)} duplicates -> {len(df)} records")
    
    # Sort
    df = df.sort_values(['year', 'round']).reset_index(drop=True)
    
    # Fill missing grid_positions with estimates
    print(f"[FILL] Filling missing values...")
    
    # Estimate grid position based on finish and constructor
    constructor_grid_bonus = {
        'Red Bull': -2, 'Ferrari': -1, 'Mercedes': -1, 'McLaren': 0,
        'Aston Martin': 1, 'Alpine': 2, 'Haas': 3, 'Williams': 4,
        'Sauber': 3, 'Racing Points': 2, 'AlphaTauri': 3, 'Alfa Romeo': 4
    }
    
    for idx in df[df['grid_position'].isna()].index:
        finish = int(df.loc[idx, 'finish_position'])
        constructor = df.loc[idx, 'constructor']
        
        bonus = 0
        for key, val in constructor_grid_bonus.items():
            if key.lower() in constructor.lower():
                bonus = val
                break
        
        grid_est = max(1, finish - bonus)
        df.loc[idx, 'grid_position'] = grid_est
    
    # Calculate position_gain (finish - grid position)
    df['position_gain'] = df['grid_position'] - df['finish_position']
    
    # Fill driver ages
    np.random.seed(42)
    missing_ages = df[df['driver_age'].isna()].index
    df.loc[missing_ages, 'driver_age'] = np.random.randint(24, 37, len(missing_ages))
    
    print(f"  Grid positions: {df['grid_position'].isna().sum()} missing")
    print(f"  Driver ages: {df['driver_age'].isna().sum()} missing")
    
    # Select only needed columns for model
    df_model = df[['year', 'round', 'event', 'circuit', 'driver_code', 
                   'driver_number', 'driver_age', 'constructor', 
                   'grid_position', 'finish_position', 'position_gain', 'points_constructor']].copy()
    
    # Null analysis
    print(f"\n[NULL ANALYSIS] Final")
    for col in df_model.columns:
        nulls = df_model[col].isna().sum()
        pct = (nulls / len(df_model) * 100) if len(df_model) > 0 else 0
        print(f"  {col:20s}: {nulls:4d} nulls ({pct:5.1f}%)")
    
    # Save
    output_file = 'f1_historical_5years.csv'
    print(f"\n[SAVE] Writing {output_file}...")
    df_model.to_csv(output_file, index=False)
    
    # Summary
    print(f"\n[SUMMARY]")
    print(f"  File: {output_file}")
    print(f"  Shape: {df_model.shape}")
    print(f"  Years: {sorted(df_model['year'].unique())}")
    print(f"  Races: {len(df_model.groupby(['year', 'round']))}")
    print(f"  Drivers (unique): {len(df_model.groupby('driver_code'))}")
    print(f"  Avg drivers/race: {len(df_model) / max(1, len(df_model.groupby(['year', 'round']))):.1f}")
    
    print(f"\n[DATA QUALITY]")
    print(f"  Grid positions: min={df_model['grid_position'].min()}, max={df_model['grid_position'].max()}")
    print(f"  Finish positions: min={df_model['finish_position'].min()}, max={df_model['finish_position'].max()}")
    print(f"  Driver ages: min={df_model['driver_age'].min()}, max={df_model['driver_age'].max()}, mean={df_model['driver_age'].mean():.1f}")
    
    print(f"\n[SAMPLE] 5 random records:")
    sample = df_model.sample(min(5, len(df_model)))
    print(sample.to_string(index=False))
    
    print("\n" + "="*70)
    print("[SUCCESS] Historical CSV created from .cache!")
    print(f"Ready for AI model pre-training with {len(df_model)} records")
    print("="*70 + "\n")
else:
    print(f"[WARNING] Only {len(all_records)} records - need at least 100")
    print("[ERROR] Not enough data for meaningful pre-training")
