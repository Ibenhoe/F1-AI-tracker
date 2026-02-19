#!/usr/bin/env python3
"""Test script to check lap data availability - why is lap 2 missing?"""

import sys
import pandas as pd
import fastf1

# Setup FastF1 cache
import os
import tempfile
cache_dir = os.path.join(tempfile.gettempdir(), 'fastf1_cache')
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

print("[TEST] Loading race 1 (Bahrain 2024)...")

try:
    session = fastf1.get_session(2024, 1, 'R')
    session.load()
    
    laps_data = session.laps
    
    if laps_data is None or len(laps_data) == 0:
        print("[TEST] ❌ No lap data available!")
    else:
        print(f"[TEST] ✓ Loaded {len(laps_data)} lap records")
        
        # Get unique lap numbers
        unique_laps = sorted(laps_data['LapNumber'].unique())
        print(f"[TEST] Unique laps: {unique_laps[:15]}...")
        
        # Check first 5 laps in detail
        for i, lap_num in enumerate(unique_laps[:5]):
            lap_records = laps_data[laps_data['LapNumber'] == lap_num]
            drivers_in_lap = lap_records['Driver'].unique()
            lap_num_int = int(lap_num)
            print(f"[TEST] Lap {lap_num_int:2d}: {len(lap_records)} records, {len(drivers_in_lap)} unique drivers")
            print(f"         Drivers: {', '.join(sorted(drivers_in_lap)[:5])}...")
            
            # Check if all required columns are present
            required_cols = ['LapNumber', 'Driver', 'Position', 'LapTime', 'Compound', 'Gap']
            missing = [col for col in required_cols if col not in lap_records.columns]
            if missing:
                print(f"         ⚠ Missing columns: {missing}")
            
            # Show available columns for first lap only
            if i == 0:
                print(f"\n[TEST] Available columns in lap data:")
                print(f"       {list(lap_records.columns)}")
                print()
        
        print("\n[TEST] Data structure looks good!")
        
except Exception as e:
    print(f"[TEST] ❌ Error: {e}")
    import traceback
    traceback.print_exc()
