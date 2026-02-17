#!/usr/bin/env python3
"""
Compare FastF1 qualifying grid with our frame data
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fastf1
from fastf1_data_fetcher import FastF1DataFetcher
from app import _build_replay_frames

def test_grid_accuracy():
    """Verify first frame matches FastF1 qualifying grid"""
    
    print("\n" + "="*70)
    print("[TEST] Grid Accuracy Check - Frame vs FastF1")
    print("="*70)
    
    # Get FastF1 qualifying data
    print("\n[FastF1] Loading qualifying data...")
    qual = fastf1.get_session(2024, 21, 'Q')
    qual.load(telemetry=False, weather=False)
    
    fastf1_grid = {}
    if qual.results is not None:
        for idx, (_, row) in enumerate(qual.results.iterrows()):
            driver_code = str(row.get('Abbreviation', ''))
            if driver_code and driver_code != 'nan':
                fastf1_grid[driver_code] = idx + 1  # 1-based position
    
    print(f"FastF1 Grid ({len(fastf1_grid)} drivers):")
    for code, pos in sorted(fastf1_grid.items(), key=lambda x: x[1])[:5]:
        print(f"  {pos:2d}. {code}")
    
    # Get our frame data
    print("\n[OUR DATA] Loading frame data...")
    fetcher = FastF1DataFetcher()
    if not fetcher.fetch_race(2024, 21):
        print("ERROR: Could not load race")
        return False
    
    laps_data = fetcher.process_race_laps_streaming()
    race_info = fetcher.get_race_summary()
    frames = _build_replay_frames(laps_data, race_info)
    
    frame_grid = {}
    if frames and len(frames) > 0:
        first_frame = frames[0]
        for code, driver in first_frame.get('drivers', {}).items():
            frame_grid[code] = driver.get('position')
    
    print(f"Frame Grid ({len(frame_grid)} drivers):")
    for code, pos in sorted(frame_grid.items(), key=lambda x: x[1])[:5]:
        print(f"  {pos:5.1f}. {code}")
    
    # Compare
    print("\n" + "-"*70)
    print("[COMPARISON] FastF1 vs Our Frame Data:")
    print("Driver | FastF1 Grid | Frame Grid | Match?")
    print("-"*70)
    
    mismatches = 0
    for code in sorted(set(fastf1_grid.keys()) | set(frame_grid.keys())):
        f1_pos = fastf1_grid.get(code, "?")
        frame_pos = frame_grid.get(code, "?")
        
        if f1_pos != "?" and frame_pos != "?":
            match = "YES" if abs(f1_pos - frame_pos) < 0.5 else "NO"
            if match == "NO":
                mismatches += 1
            print(f"{code:6s} | {f1_pos:11d} | {frame_pos:10.1f} | {match}")
    
    print("\n" + "="*70)
    if mismatches == 0:
        print("[OK] All drivers at correct starting positions!")
        return True
    else:
        print(f"[ERROR] {mismatches} drivers at wrong starting positions!")
        return False

if __name__ == "__main__":
    success = test_grid_accuracy()
    sys.exit(0 if success else 1)
