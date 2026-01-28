"""
Rebuild f1_historical_5years.csv from FastF1 cache with CLEAN data
Extracts race results from cached FastF1 sessions using .ff1pkl files
"""

import os
import sys
import tempfile
import gzip
import pickle
import pandas as pd
from pathlib import Path

print("="*70)
print("[REBUILD] F1 Historical Data CSV - Cache Extractor")
print("="*70)

# FastF1 cache location
CACHE_DIR = os.path.join(tempfile.gettempdir(), 'fastf1_cache')
OUTPUT_FILE = 'f1_historical_5years.csv'

print(f"\nCache location: {CACHE_DIR}")
print(f"Output file: {OUTPUT_FILE}\n")

if not os.path.exists(CACHE_DIR):
    print("❌ Cache directory not found!")
    sys.exit(1)

all_records = []

def load_ff1pkl(file_path):
    """Load .ff1pkl file (pickle, NOT gzipped despite the name)"""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"    [WARN] Could not load {os.path.basename(file_path)}: {e}")
        return None

# Extract data from cached sessions
print("[SCAN] Scanning cache for race data...\n")

years = [d for d in os.listdir(CACHE_DIR) if d.isdigit()]
print(f"Found years: {sorted(years)}\n")

for year in sorted(years):
    year_path = os.path.join(CACHE_DIR, year)
    
    # Get all races for this year
    races = [r for r in os.listdir(year_path) if os.path.isdir(os.path.join(year_path, r))]
    
    print(f"[YEAR {year}] Found {len(races)} races")
    
    for race_dir in sorted(races):
        race_path = os.path.join(year_path, race_dir)
        
        # Find session directories
        sessions = [s for s in os.listdir(race_path) if os.path.isdir(os.path.join(race_path, s))]
        
        for session_name in sorted(sessions):
            session_path = os.path.join(race_path, session_name)
            is_race = 'Race' in session_name
            
            if not is_race:
                continue  # Skip qualifying for now
            
            try:
                print(f"  🏁 {session_name:40s}", end=" ... ")
                
                # Try to load session_info to get metadata
                info_file = os.path.join(session_path, 'session_info.ff1pkl')
                session_info = load_ff1pkl(info_file) if os.path.exists(info_file) else {}
                
                round_num = session_info.get('RoundNumber') if isinstance(session_info, dict) else None
                event_name = session_info.get('EventName') if isinstance(session_info, dict) else 'Unknown'
                circuit = session_info.get('Circuit') if isinstance(session_info, dict) else 'Unknown'
                
                # Try to load driver_info to get driver details
                driver_info_file = os.path.join(session_path, 'driver_info.ff1pkl')
                driver_info = load_ff1pkl(driver_info_file) if os.path.exists(driver_info_file) else {}
                
                # Try to load timing_app_data which has results
                timing_file = os.path.join(session_path, 'timing_app_data.ff1pkl')
                timing_data = load_ff1pkl(timing_file) if os.path.exists(timing_file) else None
                
                record_count = 0
                
                # Check if we have driver info with finishing positions
                if driver_info and isinstance(driver_info, dict):
                    for driver_abbr, driver_details in driver_info.items():
                        try:
                            if not isinstance(driver_details, dict):
                                continue
                            
                            # Get basic info
                            driver_num = driver_details.get('DriverNumber')
                            first_name = driver_details.get('FirstName', 'Unknown')
                            last_name = driver_details.get('LastName', 'Unknown')
                            constructor = driver_details.get('TeamName', 'Unknown')
                            
                            # Try to get position from various sources
                            grid_pos = driver_details.get('GridPosition')
                            finish_pos = driver_details.get('Position')
                            
                            # Fallback: try from driver abbreviation if available
                            if finish_pos is None:
                                finish_pos = driver_details.get('FinishingPosition')
                            
                            # Skip if no finishing position
                            if finish_pos is None or (isinstance(finish_pos, (int, float)) and finish_pos > 20):
                                continue
                            
                            # Convert to int
                            try:
                                finish_pos = int(finish_pos)
                                if grid_pos:
                                    grid_pos = int(grid_pos)
                            except:
                                continue
                            
                            record = {
                                'year': int(year),
                                'round': int(round_num) if round_num else None,
                                'event': str(event_name),
                                'circuit': str(circuit),
                                'driver_code': str(driver_abbr).upper(),
                                'driver_number': int(driver_num) if driver_num else None,
                                'driver_age': None,
                                'constructor': str(constructor),
                                'grid_position': grid_pos,
                                'finish_position': finish_pos,
                                'points_constructor': 100.0,
                            }
                            
                            all_records.append(record)
                            record_count += 1
                        except Exception as e:
                            pass
                
                print(f"[OK] {record_count} drivers")
            
            except Exception as e:
                print(f"[ERROR] {str(e)[:40]}")


print(f"\n[EXTRACT] Total records extracted: {len(all_records)}")

if all_records:
    df = pd.DataFrame(all_records)
    
    # Clean data: remove duplicates
    print(f"\n[CLEAN] Data cleaning...")
    print(f"  Initial records: {len(df)}")
    
    # Remove complete duplicates
    df_clean = df.drop_duplicates(subset=['year', 'round', 'driver_code'], keep='first')
    print(f"  After removing duplicates: {len(df_clean)}")
    
    # Sort by year, round
    df_clean = df_clean.sort_values(['year', 'round']).reset_index(drop=True)
    
    # Fill NaN values
    print(f"\n[NULLS ANALYSIS]")
    for col in df_clean.columns:
        nulls = df_clean[col].isna().sum()
        if nulls > 0:
            pct = (nulls / len(df_clean) * 100)
            print(f"  {col:20s}: {nulls:4d} nulls ({pct:5.1f}%)")
    
    # Save to CSV
    print(f"\n[SAVE] Writing {OUTPUT_FILE}...")
    df_clean.to_csv(OUTPUT_FILE, index=False)
    print(f"[OK] Saved {len(df_clean)} records")
    
    # Summary
    print(f"\n[SUMMARY] CSV Structure:")
    print(f"  Shape: {df_clean.shape}")
    print(f"  Columns: {list(df_clean.columns)}")
    print(f"  Years: {sorted(df_clean['year'].unique())}")
    print(f"  Total races: {len(df_clean.groupby(['year', 'round']))}")
    print(f"  Total drivers: {len(df_clean.groupby('driver_code'))}")
    
    print(f"\n[SAMPLE] First 5 records:")
    print(df_clean.head().to_string())
else:
    print("[ERROR] No records extracted!")

print("\n" + "="*70)
print("[DONE] Rebuild complete!")
print("="*70 + "\n")

