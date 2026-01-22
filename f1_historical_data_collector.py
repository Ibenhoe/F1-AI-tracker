"""
F1 HISTORICAL DATA COLLECTOR
Collect 5 years of F1 data (2020-2024) including qualifying + race results
Used for pre-training the AI model with realistic historical patterns
"""

import fastf1
import pandas as pd
import numpy as np
from datetime import datetime
import os
import time

# Create cache directory if it doesn't exist
cache_dir = '.cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
    print(f"[OK] Created cache directory: {cache_dir}")

# Enable cache to avoid re-downloading
fastf1.Cache.enable_cache(cache_dir=cache_dir)

# Configuration
SEASONS = [2024, 2025]
OUTPUT_FILE = 'f1_historical_5years.csv'

print("="*70)
print("[F1-DATA] F1 HISTORICAL DATA COLLECTOR (2024-2025 APPEND)")
print("="*70)

# Load existing data if it exists
all_races_data = []
if os.path.exists(OUTPUT_FILE):
    print(f"[OK] Loading existing data from {OUTPUT_FILE}")
    existing_df = pd.read_csv(OUTPUT_FILE)
    all_races_data = existing_df.to_dict('records')
    print(f"  Found {len(all_races_data)} existing records")
else:
    print(f"[INFO] No existing file found, starting fresh")

for season in SEASONS:
    print(f"\n[SEASON] Processing {season}...")
    
    try:
        # Get schedule for the season
        schedule = fastf1.get_event_schedule(season)
        print(f"  Found {len(schedule)} races in {season}")
        
        if len(schedule) == 0:
            print(f"  [WARN] No races found for {season}")
            continue
        
        for idx, race in schedule.iterrows():
            if pd.isna(race['RoundNumber']):
                continue
                
            round_num = int(race['RoundNumber'])
            event_name = race['EventName']
            
            try:
                print(f"    [{round_num:2d}] {event_name:30s}", end=" ... ")
                
                # Load session with better error handling
                try:
                    session = fastf1.get_session(season, round_num, 'R')
                    session.load(telemetry=False, weather=False)
                except Exception as e:
                    print(f"[SKIP - Race load failed]")
                    continue
                
                # Get qualifying session for grid info
                qual_results = None
                try:
                    qual_session = fastf1.get_session(season, round_num, 'Q')
                    qual_session.load(telemetry=False, weather=False)
                    qual_results = qual_session.results[['DriverNumber', 'GridPosition']].copy()
                except:
                    pass  # Qualifying data optional
                
                # Get race results
                results = session.results.copy()
                
                if len(results) == 0:
                    print(f"[SKIP - No results]")
                    continue
                
                # Process each driver
                for driver_idx, driver in results.iterrows():
                    driver_number = driver['DriverNumber']
                    
                    try:
                        # Get grid position
                        if qual_results is not None:
                            qual_row = qual_results[qual_results['DriverNumber'] == driver_number]
                            if not qual_row.empty:
                                grid_pos = int(qual_row.iloc[0]['GridPosition'])
                            else:
                                grid_pos = int(driver['GridPosition']) if pd.notna(driver['GridPosition']) else 20
                        else:
                            grid_pos = int(driver['GridPosition']) if pd.notna(driver['GridPosition']) else 20
                        
                        # Get finish position
                        finish_pos = int(driver['Position']) if pd.notna(driver['Position']) else 20
                        
                        # Get points and other info
                        points = float(driver['Points']) if pd.notna(driver['Points']) else 0
                        dnf = 1 if pd.isna(driver['Position']) else 0
                        
                        # Get driver info
                        driver_code = driver['DriverCode'] if pd.notna(driver['DriverCode']) else 'UNK'
                        constructor = driver['Constructor'] if pd.notna(driver['Constructor']) else 'UNK'
                        
                        race_data = {
                            'year': season,
                            'round': round_num,
                            'event': event_name,
                            'driver_number': driver_number,
                            'driver_code': driver_code,
                            'constructor': constructor,
                            'grid_position': grid_pos,
                            'finish_position': finish_pos,
                            'position_gain': grid_pos - finish_pos,  # Positive = gained places
                            'points': points,
                            'dnf': dnf,
                            'date': race['EventDate']
                        }
                        
                        all_races_data.append(race_data)
                    
                    except Exception as e:
                        continue
                
                print(f"[OK]")
                time.sleep(0.5)  # Polite delay
                
            except Exception as e:
                print(f"[ERROR] {str(e)[:50]}")
                continue
    
    except Exception as e:
        print(f"[WARN] Season {season}: {e}")
        continue

print(f"\n[DEBUG] Total records collected: {len(all_races_data)}")

# Create DataFrame
if all_races_data:
    print("[INFO] Creating DataFrame from collected records...")
    df = pd.DataFrame(all_races_data)
    print(f"  DataFrame created: {len(df)} rows")
    
    # Save to CSV
    print(f"[INFO] Saving to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[OK] Data saved to: {OUTPUT_FILE}")
    print(f"    Total records: {len(df)}")
    print(f"    Columns: {df.columns.tolist()}")
    if len(df) > 0:
        print(f"    Years: {df['year'].min()}-{df['year'].max()}")
        print(f"    Max round: {df['round'].max()}")
        
        # Show summary
        print("\n[SUMMARY] Data Statistics:")
        races = df.groupby(['year', 'round']).size()
        print(f"  - Total races: {len(races)}")
        print(f"  - Drivers per race: {races.mean():.1f}")
        print(f"  - Avg position gain: {df['position_gain'].mean():.2f}")
        print(f"  - DNF rate: {df['dnf'].mean()*100:.1f}%")
        print(f"  - Position range: {df['finish_position'].min():.0f}-{df['finish_position'].max():.0f}")
        
        # Show sample
        print("\n[SAMPLE] First 5 rows:")
        print(df.head())
else:
    print("[ERROR] No data collected!")

print("\n" + "="*70)
print("[OK] Data collection complete!")
print(f"[CHECK] File exists: {os.path.exists(OUTPUT_FILE)}")
print("="*70)
