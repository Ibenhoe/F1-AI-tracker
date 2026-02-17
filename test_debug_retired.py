#!/usr/bin/env python3
"""
Debug test to trace why retired drivers disappear from frames
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastf1_data_fetcher import FastF1DataFetcher

def test_debug():
    """Debug retired driver frame generation"""
    
    print("\n" + "="*70)
    print("[DEBUG] Checking lap data for retired drivers")
    print("="*70)
    
    # Load race data
    print("\n[LOAD] Loading race data...")
    fetcher = FastF1DataFetcher()
    if not fetcher.fetch_race(2024, 21):
        print("ERROR: Could not load race")
        return False
    
    laps_data = fetcher.process_race_laps_streaming()
    
    # Group by lap number
    laps_by_number = {}
    for lap in laps_data:
        lap_num = int(lap.get('lap_number', 0))
        if lap_num not in laps_by_number:
            laps_by_number[lap_num] = []
        laps_by_number[lap_num].append(lap)
    
    sorted_laps = sorted(laps_by_number.keys())
    
    # Check SAI's data
    print("\n[SAI Analysis]:")
    sai_laps = set()
    for lap_num in sorted_laps:
        for lap_record in laps_by_number[lap_num]:
            if lap_record.get('driver') == 'SAI':
                sai_laps.add(lap_num)
    
    sai_laps_sorted = sorted(sai_laps)
    print(f"  SAI completed laps: {min(sai_laps_sorted)} to {max(sai_laps_sorted)}")
    print(f"  Last lap with SAI data: {max(sai_laps_sorted)}")
    print(f"  Frames that should have SAI: 0 to {max(sai_laps_sorted) * 30 + 29}")
    
    # Check HUL's data
    print("\n[HUL Analysis]:")
    hul_laps = set()
    for lap_num in sorted_laps:
        for lap_record in laps_by_number[lap_num]:
            if lap_record.get('driver') == 'HUL':
                hul_laps.add(lap_num)
    
    hul_laps_sorted = sorted(hul_laps)
    print(f"  HUL completed laps: {min(hul_laps_sorted)} to {max(hul_laps_sorted)}")
    print(f"  Last lap with HUL data: {max(hul_laps_sorted)}")
    print(f"  Frames that should have HUL: 0 to {max(hul_laps_sorted) * 30 + 29}")
    
    # Check if drivers appear in laps after their last recorded lap
    print("\n[CHECK] Do retired drivers have ANY data in laps after they retired?")
    for lap_num in sorted_laps[-10:]:  # Check last 10 laps
        drivers_this_lap = set(lap.get('driver') for lap in laps_by_number[lap_num])
        if 'SAI' in drivers_this_lap:
            print(f"  SAI found in lap {lap_num}")
        if 'HUL' in drivers_this_lap:
            print(f"  HUL found in lap {lap_num}")
    
    print("\n[CONCLUSION]")
    print("Retired drivers have NO data after their last lap.")
    print("The frame generation should use last_driver_state to keep them visible.")
    print("But our implementation only shows them until their last lap.")
    print("\nThis is because:")
    print("1. We check: if not current_record -> use last_driver_state")
    print("2. But we also check: if not current_record AND not prev_record -> continue")
    print("3. So if both are missing, the driver gets skipped entirely.")
    print("\nThe fix: Always use last_driver_state if available, never skip it.")

if __name__ == "__main__":
    test_debug()
