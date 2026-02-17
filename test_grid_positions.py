#!/usr/bin/env python3
"""
Test to check if drivers are at correct grid positions in first frame
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastf1_data_fetcher import FastF1DataFetcher
from app import _build_replay_frames

def test_first_frame_positions():
    """Check if first frame has drivers at grid positions"""
    
    print("\n" + "="*70)
    print("[TEST] First Frame Grid Position Check")
    print("="*70)
    
    # Load race data
    print("\n[LOAD] Loading race data...")
    fetcher = FastF1DataFetcher()
    if not fetcher.fetch_race(2024, 21):
        print("ERROR: Could not load race")
        return False
    
    laps_data = fetcher.process_race_laps_streaming()
    race_info = fetcher.get_race_summary()
    
    # Build frames
    print("[BUILD] Building frames...")
    frames = _build_replay_frames(laps_data, race_info)
    
    if not frames or len(frames) == 0:
        print("ERROR: No frames generated")
        return False
    
    # Check first frame
    first_frame = frames[0]
    drivers = first_frame.get('drivers', {})
    
    print(f"\n[FIRST FRAME] Lap {first_frame.get('lap')}, Frame 0")
    print(f"Total drivers: {len(drivers)}")
    
    # Sort by position
    drivers_list = [(code, driver) for code, driver in drivers.items()]
    drivers_list.sort(key=lambda x: x[1].get('position', 999))
    
    print("\nDriver Grid Positions (should be 1-18):")
    print("-" * 70)
    for code, driver in drivers_list[:10]:
        pos = driver.get('position')
        name = driver.get('driver_name', 'Unknown')
        print(f"  {code}: Position {pos:.1f} - {name}")
    
    # Check if positions are sequential
    print("\n[CHECK] Are positions sequential 1-18?")
    positions = sorted([driver.get('position', 999) for driver in drivers.values() if driver.get('position') is not None])
    
    if len(positions) == len(drivers):
        print(f"  Positions: {[f'{p:.0f}' for p in positions]}")
        
        # Check if they're close to 1,2,3,...,18
        expected = list(range(1, len(positions) + 1))
        all_correct = all(abs(p - e) < 0.5 for p, e in zip(sorted(positions), expected))
        
        if all_correct:
            print("  [OK] All drivers at correct grid positions!")
            return True
        else:
            print("  [ERROR] Positions don't match grid (1-18)!")
            return False
    else:
        print(f"  [ERROR] Only {len(positions)} drivers have positions (expected {len(drivers)})")
        return False

if __name__ == "__main__":
    success = test_first_frame_positions()
    sys.exit(0 if success else 1)
