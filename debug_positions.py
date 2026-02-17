#!/usr/bin/env python3
"""
Debug why drivers' positions change unexpectedly
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastf1_data_fetcher import FastF1DataFetcher

def debug_positions():
    """Check actual lap data for selected driver"""
    
    print("\n" + "="*70)
    print("[DEBUG] Checking raw lap data for position changes")
    print("="*70)
    
    fetcher = FastF1DataFetcher()
    if not fetcher.fetch_race(2024, 21):
        print("ERROR: Could not load race")
        return False
    
    laps_data = fetcher.process_race_laps_streaming()
    
    # Group by lap
    laps_by_number = {}
    for lap in laps_data:
        lap_num = int(lap.get('lap_number', 0))
        if lap_num not in laps_by_number:
            laps_by_number[lap_num] = []
        laps_by_number[lap_num].append(lap)
    
    sorted_laps = sorted(laps_by_number.keys())
    
    # Check RUS's positions
    print("\n[RUS Positions Across Laps]")
    print("Lap | Position")
    print("----|----------")
    
    rus_prev_pos = None
    for lap_num in sorted_laps:
        for lap_record in laps_by_number[lap_num]:
            if lap_record.get('driver') == 'RUS':
                pos = lap_record.get('position')
                pit_status = lap_record.get('pit_stops', 0)
                
                change = ""
                if rus_prev_pos is not None:
                    if pos > rus_prev_pos:
                        change = f" <- LOSS ({rus_prev_pos} to {pos})"
                    elif pos < rus_prev_pos:
                        change = f" <- GAIN ({rus_prev_pos} to {pos})"
                
                print(f"{lap_num:3d} | {pos:8.0f}{change}")
                rus_prev_pos = pos
                break
    
    # Focus on the lap where RUS goes from P1 to P4
    print("\n[DETAILED] Checking laps 30-40 for all drivers:")
    print("-" * 70)
    
    for lap_num in range(30, 41):
        if lap_num not in laps_by_number:
            continue
        
        lap_drivers = {}
        for lap_record in laps_by_number[lap_num]:
            driver = lap_record.get('driver')
            pos = lap_record.get('position')
            if driver and pos:
                lap_drivers[driver] = pos
        
        print(f"\nLap {lap_num}: {', '.join(f'{d}=P{int(p)}' for d, p in sorted(lap_drivers.items(), key=lambda x: x[1]))}")

if __name__ == "__main__":
    debug_positions()
