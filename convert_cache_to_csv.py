"""
Convert FastF1 .ff1pkl cache files to CSV format
Reads cached session data and extracts qualifying + race results
"""

import os
import pickle
import gzip
import glob
import pandas as pd
from pathlib import Path

OUTPUT_FILE = 'f1_historical_5years.csv'
CACHE_DIR = '.cache'

print("="*70)
print("[CONVERTER] FastF1 Cache (.ff1pkl) to CSV Converter")
print("="*70)

all_records = []

# Find all .ff1pkl files
pkl_files = glob.glob(os.path.join(CACHE_DIR, '**', '*.ff1pkl'), recursive=True)
print(f"\n[INFO] Found {len(pkl_files)} .ff1pkl cache files\n")

for pkl_file in sorted(pkl_files):
    try:
        rel_path = os.path.relpath(pkl_file, CACHE_DIR)
        print(f"[READ] {rel_path:60s}", end=" ... ")
        
        # Try to read pickle file (try both gzip and plain pickle)
        try:
            with gzip.open(pkl_file, 'rb') as f:
                data = pickle.load(f)
        except:
            # Try plain pickle if gzip fails
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
        
        # Try to extract session results
        if hasattr(data, 'results') and data.results is not None and len(data.results) > 0:
            results_df = data.results.copy()
            
            # Try to get metadata from the session
            year = None
            round_num = None
            event_name = None
            session_type = None
            
            # Extract metadata from path (typical format: year/round_num/...)
            path_parts = rel_path.split(os.sep)
            if len(path_parts) >= 2:
                try:
                    year = int(path_parts[0])
                    round_num = int(path_parts[1].split('_')[0])
                except:
                    pass
            
            # Try to extract from session object
            if hasattr(data, 'session_info'):
                try:
                    if 'Year' in data.session_info:
                        year = data.session_info['Year']
                    if 'RoundNumber' in data.session_info:
                        round_num = data.session_info['RoundNumber']
                    if 'EventName' in data.session_info:
                        event_name = data.session_info['EventName']
                    if 'SessionType' in data.session_info:
                        session_type = data.session_info['SessionType']
                except:
                    pass
            
            # Process each driver result
            for driver_idx, driver in results_df.iterrows():
                try:
                    record = {
                        'year': year,
                        'round': round_num,
                        'event': event_name or 'Unknown',
                        'session_type': session_type or 'Unknown',
                        'driver_number': int(driver['DriverNumber']) if pd.notna(driver['DriverNumber']) else None,
                        'driver_code': driver['DriverCode'] if pd.notna(driver['DriverCode']) else 'UNK',
                        'first_name': driver['FirstName'] if pd.notna(driver['FirstName']) else '',
                        'last_name': driver['LastName'] if pd.notna(driver['LastName']) else '',
                        'constructor': driver['Constructor'] if pd.notna(driver['Constructor']) else 'UNK',
                        'points': float(driver['Points']) if pd.notna(driver['Points']) else 0,
                        'grid_position': int(driver['GridPosition']) if pd.notna(driver['GridPosition']) else None,
                        'position': int(driver['Position']) if pd.notna(driver['Position']) else None,
                        'status': driver['Status'] if pd.notna(driver['Status']) else 'Unknown',
                    }
                    
                    # Calculate position gain (grid -> finish)
                    if record['grid_position'] is not None and record['position'] is not None:
                        record['position_gain'] = record['grid_position'] - record['position']
                    else:
                        record['position_gain'] = 0
                    
                    all_records.append(record)
                except Exception as e:
                    pass
            
            print(f"[OK] {len(results_df)} drivers")
        else:
            print(f"[SKIP] No results")
    
    except Exception as e:
        print(f"[ERROR] {str(e)[:50]}")

print(f"\n[INFO] Total records extracted: {len(all_records)}")

if all_records:
    df = pd.DataFrame(all_records)
    
    # Merge with existing data if it exists
    if os.path.exists('processed_f1_training_data.csv'):
        print(f"[INFO] Loading existing data...")
        existing_df = pd.read_csv('processed_f1_training_data.csv')
        print(f"  Existing records: {len(existing_df)}")
        
        # Combine and remove duplicates
        df = pd.concat([existing_df, df], ignore_index=True)
        df = df.drop_duplicates(subset=['year', 'round', 'driver_number', 'session_type'], keep='first')
        print(f"  Combined records: {len(df)}")
    
    # Save to CSV
    print(f"\n[SAVE] Writing {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[OK] Saved {len(df)} records")
    print(f"  Columns: {df.columns.tolist()}")
    
    if 'year' in df.columns:
        years = df['year'].dropna().unique()
        print(f"  Years covered: {sorted(years)}")
        print(f"  Sessions: {df['session_type'].unique().tolist()}")
    
    print("\n[SAMPLE] First 5 rows:")
    print(df.head())
else:
    print("[ERROR] No records extracted!")

print("\n" + "="*70)
print("[OK] Conversion complete!")
