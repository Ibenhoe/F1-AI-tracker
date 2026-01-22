"""
Direct F1 Data Extractor - Pulls 2024/2025 data via FastF1 API
Merges with existing processed_f1_training_data.csv
"""

import fastf1
import pandas as pd
import os

OUTPUT_FILE = 'f1_historical_5years.csv'
EXISTING_FILE = 'processed_f1_training_data.csv'
CACHE_DIR = '.cache'

# Enable cache
fastf1.Cache.enable_cache(cache_dir=CACHE_DIR)

print("="*70)
print("[EXTRACTOR] F1 2024/2025 Data Extractor")
print("="*70)

# Load existing data
all_records = []
if os.path.exists(EXISTING_FILE):
    print(f"\n[LOAD] Loading existing data from {EXISTING_FILE}...")
    existing_df = pd.read_csv(EXISTING_FILE)
    print(f"  Found {len(existing_df)} existing records")
    all_records = existing_df.to_dict('records')

new_records = 0

# Get 2024 and 2025 data
for year in [2024, 2025]:
    print(f"\n[YEAR] Processing {year}...")
    
    try:
        schedule = fastf1.get_event_schedule(year)
        print(f"  Found {len(schedule)} races")
        
        for idx, race in schedule.iterrows():
            if pd.isna(race['RoundNumber']):
                continue
            
            round_num = int(race['RoundNumber'])
            event_name = race['EventName']
            
            # Skip if already in data
            existing_check = [r for r in all_records if r.get('year') == year and r.get('round') == round_num]
            if existing_check:
                print(f"  [{round_num:2d}] {event_name:30s} [SKIP - already exists]")
                continue
            
            print(f"  [{round_num:2d}] {event_name:30s}", end=" ... ")
            
            try:
                # Load race session
                session = fastf1.get_session(year, round_num, 'R')
                session.load(telemetry=False, weather=False)
                results = session.results.copy()
                
                if len(results) == 0:
                    print("[SKIP - no results]")
                    continue
                
                # Load qualifying for grid info
                try:
                    qual_session = fastf1.get_session(year, round_num, 'Q')
                    qual_session.load(telemetry=False, weather=False)
                    qual_results = qual_session.results[['DriverNumber', 'GridPosition']].copy()
                except:
                    qual_results = None
                
                # Extract each driver
                for driver_idx, driver in results.iterrows():
                    try:
                        driver_number = int(driver['DriverNumber'])
                        
                        # Get grid position
                        if qual_results is not None:
                            qual_row = qual_results[qual_results['DriverNumber'] == driver_number]
                            grid_pos = int(qual_row.iloc[0]['GridPosition']) if not qual_row.empty else int(driver['GridPosition']) if pd.notna(driver['GridPosition']) else 20
                        else:
                            grid_pos = int(driver['GridPosition']) if pd.notna(driver['GridPosition']) else 20
                        
                        # Get finish position
                        finish_pos = int(driver['Position']) if pd.notna(driver['Position']) else 20
                        
                        # Get driver code - handle missing field
                        driver_code = 'UNK'
                        if 'DriverCode' in driver and pd.notna(driver['DriverCode']):
                            driver_code = driver['DriverCode']
                        elif 'Abbreviation' in driver and pd.notna(driver['Abbreviation']):
                            driver_code = driver['Abbreviation']
                        
                        record = {
                            'year': year,
                            'round': round_num,
                            'event': event_name,
                            'driver_number': driver_number,
                            'driver_code': driver_code,
                            'constructor': driver['Constructor'] if 'Constructor' in driver and pd.notna(driver['Constructor']) else 'UNK',
                            'grid_position': grid_pos,
                            'finish_position': finish_pos,
                            'position_gain': grid_pos - finish_pos,
                            'points': float(driver['Points']) if pd.notna(driver['Points']) else 0,
                            'dnf': 1 if pd.isna(driver['Position']) else 0,
                        }
                        
                        all_records.append(record)
                        new_records += 1
                    except Exception as driver_err:
                        pass  # Skip this driver
                
                
                print(f"[OK] {len(results)} drivers")
            
            except Exception as e:
                print(f"[ERROR] {str(e)[:40]}")
    
    except Exception as e:
        print(f"  [ERROR] {str(e)[:50]}")

print(f"\n[RESULT] Added {new_records} new records")

if all_records:
    df = pd.DataFrame(all_records)
    
    # Remove exact duplicates only if columns exist
    if 'year' in df.columns and 'round' in df.columns and 'driver_number' in df.columns:
        df = df.drop_duplicates(subset=['year', 'round', 'driver_number'], keep='first')
    
    # Save
    print(f"\n[SAVE] Saving {len(df)} total records to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n[SUMMARY]")
    print(f"  Total records: {len(df)}")
    print(f"  Years: {sorted(df['year'].unique())}")
    print(f"  Drivers per race: {df.groupby(['year', 'round']).size().mean():.1f}")
    print(f"  Avg position gain: {df['position_gain'].mean():.2f}")
    print(f"  DNF rate: {df['dnf'].mean()*100:.1f}%")
    
    print(f"\n[OK] Complete!")
else:
    print("[ERROR] No records collected!")
