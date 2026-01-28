"""
Generate f1_historical_5years.csv from FastF1 cache pickle files directly
No API calls - just extract what's already cached
"""

import os
import tempfile
import pickle
import pandas as pd
import sys
from pathlib import Path

print("="*70)
print("[GENERATOR] F1 Historical CSV from FastF1 Cache (Direct Pickle)")
print("="*70)

cache_dir = os.path.join(tempfile.gettempdir(), 'fastf1_cache')
print(f"\nCache location: {cache_dir}")

if not os.path.exists(cache_dir):
    print("ERROR: Cache directory not found!")
    sys.exit(1)

all_records = []

# Scan all years in cache
for year_dir in sorted(os.listdir(cache_dir)):
    year_path = os.path.join(cache_dir, year_dir)
    
    if not os.path.isdir(year_path) or not year_dir.isdigit():
        continue
    
    year = int(year_dir)
    print(f"\n[YEAR {year}]")
    
    # Get all races for this year
    races = [r for r in os.listdir(year_path) if os.path.isdir(os.path.join(year_path, r))]
    print(f"  Found {len(races)} races")
    
    for race_dir in sorted(races):
        race_path = os.path.join(year_path, race_dir)
        
        # Get session directories
        sessions = [s for s in os.listdir(race_path) if os.path.isdir(os.path.join(race_path, s))]
        
        for session_name in sorted(sessions):
            session_path = os.path.join(race_path, session_name)
            
            # Only process Race sessions for finish positions
            if 'Race' not in session_name:
                continue
            
            try:
                print(f"    {session_name:50s}", end=" ... ")
                
                # Load session_info to get metadata
                session_info_file = os.path.join(session_path, 'session_info.ff1pkl')
                session_info = {}
                if os.path.exists(session_info_file):
                    try:
                        with open(session_info_file, 'rb') as f:
                            session_info = pickle.load(f)
                    except:
                        session_info = {}
                
                # Try to extract event info from session_info 
                if isinstance(session_info, dict):
                    round_num = session_info.get('RoundNumber')
                    event_name = session_info.get('EventName', 'Unknown')
                    circuit_obj = session_info.get('Circuit', {})
                    if isinstance(circuit_obj, dict):
                        circuit = circuit_obj.get('CircuitName', 'Unknown')
                    else:
                        circuit = str(circuit_obj) if circuit_obj else 'Unknown'
                else:
                    # Fallback: try to parse from directory name
                    round_num = None
                    event_name = race_dir.replace('_', ' ')
                    circuit = 'Unknown'
                
                # Load driver_info to get drivers
                driver_info_file = os.path.join(session_path, 'driver_info.ff1pkl')
                drivers_data = {}
                if os.path.exists(driver_info_file):
                    try:
                        with open(driver_info_file, 'rb') as f:
                            driver_info_obj = pickle.load(f)
                            if isinstance(driver_info_obj, dict) and 'data' in driver_info_obj:
                                drivers_data = driver_info_obj['data']
                    except:
                        drivers_data = {}
                
                # Extract position from driver line number (Line 1 = Position 1, etc)
                driver_count = 0
                for driver_abbr, driver_details in drivers_data.items():
                    try:
                        if not isinstance(driver_details, dict):
                            continue
                        
                        # Line field represents finishing position (1-based)
                        finish_pos = driver_details.get('Line')
                        driver_number = driver_details.get('RacingNumber')
                        team_name = driver_details.get('TeamName', 'Unknown')
                        
                        # Skip if no position or invalid
                        if finish_pos is None or not isinstance(finish_pos, (int, float)):
                            continue
                        
                        finish_pos = int(finish_pos)
                        if finish_pos > 20 or finish_pos < 1:
                            continue
                        
                        # We don't have grid position in driver_info, skip for now
                        # (would need to load from laps or timing data)
                        
                        record = {
                            'year': year,
                            'round': int(round_num) if round_num else None,
                            'event': str(event_name),
                            'circuit': str(circuit),
                            'driver_code': str(driver_abbr).upper(),
                            'driver_number': int(driver_number) if driver_number else None,
                            'driver_age': None,
                            'constructor': str(team_name),
                            'grid_position': None,  # Not available in driver_info
                            'finish_position': finish_pos,
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
        if nulls > 0:
            pct = (nulls / len(df) * 100)
            print(f"  {col:20s}: {nulls:4d} nulls ({pct:5.1f}%)")
    
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
    print(f"  Avg drivers per race: {len(df) / len(df.groupby(['year', 'round'])):.1f}")
    
    print(f"\n[SAMPLE] First 15 records:")
    print(df.head(15).to_string())
    
    print("\n" + "="*70)
    print("[SUCCESS] CSV generated from cache!")
    print("="*70 + "\n")
else:
    print("[ERROR] No data extracted!")
    sys.exit(1)
