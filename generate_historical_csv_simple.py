"""
Generate f1_historical_5years.csv for recent years only (2023-2024)
Using cached data from FastF1 that we know exists
"""

import fastf1
import pandas as pd
import sys

print("="*70)
print("[GENERATOR] F1 Historical Data CSV - Recent Years Only")
print("="*70)

# Only try recent years that we know have cache
years = [2023, 2024]

all_records = []
error_count = 0

for year in years:
    print(f"\n[YEAR {year}]")
    
    # Get schedule for the year
    try:
        schedule = fastf1.get_event_schedule(year)
        print(f"  OK Loaded schedule: {len(schedule)} races")
    except Exception as e:
        print(f"  ERROR Could not load schedule: {str(e)[:50]}")
        continue
    
    for idx, (_, event_row) in enumerate(schedule.iterrows()):
        try:
            round_num = int(event_row['RoundNumber'])
            event_name = event_row['EventName']
            circuit = 'Unknown'
            if 'Circuit' in event_row.index and pd.notna(event_row['Circuit']):
                circuit = event_row['Circuit']
            
            print(f"    [{round_num:2d}] {event_name:30s}", end=" ... ")
            
            # Load RACE session
            session = fastf1.get_session(year, round_num, 'R')
            session.load()
            
            if session.results is None or len(session.results) == 0:
                print("[SKIP] No race data")
                continue
            
            race_data = session.results
            driver_count = 0
            
            # Extract each driver
            for driver_idx, driver_row in race_data.iterrows():
                try:
                    driver_code = driver_row.get('DriverCode', 'UNK')
                    driver_number = driver_row.get('DriverNumber')
                    
                    if pd.isna(driver_code) or driver_code == 'NaN' or driver_code == '':
                        continue
                    
                    grid_pos = driver_row.get('GridPosition')
                    finish_pos = driver_row.get('Position')
                    
                    try:
                        if pd.notna(grid_pos):
                            grid_pos = int(grid_pos)
                        else:
                            grid_pos = None
                            
                        if pd.notna(finish_pos):
                            finish_pos = int(finish_pos)
                        else:
                            finish_pos = None
                    except:
                        continue
                    
                    if finish_pos is None or finish_pos > 20:
                        continue
                    
                    constructor = driver_row.get('Constructor', 'Unknown')
                    if pd.isna(constructor):
                        constructor = 'Unknown'
                    
                    record = {
                        'year': int(year),
                        'round': int(round_num),
                        'event': str(event_name),
                        'circuit': str(circuit),
                        'driver_code': str(driver_code).upper(),
                        'driver_number': int(driver_number) if pd.notna(driver_number) else None,
                        'driver_age': None,
                        'constructor': str(constructor),
                        'grid_position': grid_pos,
                        'finish_position': finish_pos,
                        'points_constructor': 100.0,
                        'dnf': 'DNF' in str(driver_row.get('Status', '')),
                    }
                    
                    all_records.append(record)
                    driver_count += 1
                    
                except Exception as e:
                    pass
            
            print(f"[OK] {driver_count} drivers")
            
        except Exception as e:
            error_count += 1
            print(f"[ERROR] {str(e)[:40]}")

print(f"\n[EXTRACT] Total records: {len(all_records)}")
print(f"[ERRORS] Failed races: {error_count}\n")

if all_records:
    # Create DataFrame
    df = pd.DataFrame(all_records)
    
    # Clean duplicates
    print("[CLEAN] Removing duplicates...")
    initial_count = len(df)
    df = df.drop_duplicates(subset=['year', 'round', 'driver_code'], keep='first')
    print(f"  Removed {initial_count - len(df)} duplicates")
    
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
    print(f"  Drivers: {len(df.groupby('driver_code'))}")
    
    print(f"\n[SAMPLE] First 10 records:")
    print(df.head(10).to_string())
    
    print("\n" + "="*70)
    print("[SUCCESS] CSV generated!")
    print("="*70 + "\n")
else:
    print("[ERROR] No data extracted!")
    sys.exit(1)
