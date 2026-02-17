#!/usr/bin/env python3
"""Test script to verify driver code mapping fix"""

from fastf1_data_fetcher import FastF1DataFetcher

print("[TEST] Loading race 21 (Abu Dhabi) data...")
fetcher = FastF1DataFetcher()
result = fetcher.fetch_race(2024, 21)

if result:
    print("[OK] Race loaded")
    laps_data = fetcher.process_race_laps_streaming()
    
    if laps_data:
        print(f"[OK] {len(laps_data)} laps processed")
        
        # Show first few laps with driver codes
        print("\nFirst 10 laps:")
        print("-" * 80)
        for i, lap in enumerate(laps_data[:10]):
            driver = lap.get('driver')
            lap_num = lap.get('lap_number')
            pos = lap.get('position')
            x = lap.get('x')
            y = lap.get('y')
            print(f"  Lap {lap_num:2.0f}: {driver:3s} P{pos:2.0f} | x={x} y={y}")
        
        # Count unique drivers
        drivers = set(lap.get('driver') for lap in laps_data)
        print(f"\nUnique drivers in race: {sorted(drivers)}")
        print(f"Total: {len(drivers)} drivers")
else:
    print("[ERROR] Could not load race")
