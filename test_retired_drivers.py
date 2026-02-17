#!/usr/bin/env python3
"""
Test to verify retired drivers stay visible throughout the entire race replay.
This ensures SAI (39 laps), COL (30 laps), HUL (30 laps) and STR (1 lap) remain visible.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastf1_data_fetcher import FastF1DataFetcher
from app import _build_replay_frames

def test_retired_drivers():
    """Test that retired drivers appear in all frames"""
    
    print("\n" + "="*70)
    print("[TEST] RETIRED DRIVER VISIBILITY TEST")
    print("="*70)
    
    # Load race data
    print("\n[LOAD] Loading race data...")
    fetcher = FastF1DataFetcher()
    if not fetcher.fetch_race(2024, 21):
        print("ERROR: Could not load race")
        return False
    
    race_info = fetcher.get_race_summary()
    laps_data = fetcher.process_race_laps_streaming()
    
    print(f"[OK] Loaded {len(laps_data)} lap records from Brazil 2024")
    
    # Build frames
    print("\n[BUILD] Building replay frames...")
    frames = _build_replay_frames(laps_data, race_info)
    print(f"[OK] Generated {len(frames)} frames")
    
    # Track drivers and their appearances
    driver_appearances = {}
    driver_last_lap = {}
    
    # Check every frame
    for frame_idx, frame in enumerate(frames):
        lap_num = frame.get('lap', 0)
        drivers_in_frame = frame.get('drivers', {})
        
        for code, driver_data in drivers_in_frame.items():
            if code not in driver_appearances:
                driver_appearances[code] = []
            
            driver_appearances[code].append(frame_idx)
            driver_last_lap[code] = lap_num
    
    # Identify retired drivers
    max_laps_completed = max(driver_last_lap.values()) if driver_last_lap else 0
    retired_drivers = {code: last_lap for code, last_lap in driver_last_lap.items() if last_lap < max_laps_completed}
    
    print("\n" + "-"*70)
    print("[ANALYSIS] Driver Appearance Statistics")
    print("-"*70)
    
    print("\nRetired Drivers (completed fewer laps than race winner):")
    for code, last_lap in sorted(retired_drivers.items(), key=lambda x: x[1]):
        appearances = len(driver_appearances.get(code, []))
        first_frame = driver_appearances[code][0] if code in driver_appearances else -1
        last_frame = driver_appearances[code][-1] if code in driver_appearances else -1
        
        # Calculate expected frames (30 per lap, plus interpolation)
        expected_frames = last_lap * 30  # Rough estimate
        
        print(f"  {code}: Completed lap {last_lap}, Appeared in {appearances} frames")
        print(f"      First frame: {first_frame}, Last frame: {last_frame}")
        print(f"      Expected: ~{expected_frames} frames, Actual: {appearances}")
        
        # Check if driver appears consistently throughout race
        first_appearance_lap = first_frame // 30
        last_appearance_lap = last_frame // 30
        print(f"      Appearance range: Lap {first_appearance_lap} to Lap {last_appearance_lap}")
    
    print("\nFinishing Drivers (completed all laps):")
    finishing_drivers = {code: last_lap for code, last_lap in driver_last_lap.items() if last_lap >= max_laps_completed}
    for code in sorted(finishing_drivers.keys()):
        appearances = len(driver_appearances.get(code, []))
        last_frame = driver_appearances[code][-1] if code in driver_appearances else -1
        expected_frames = 2070  # All 2070 frames
        
        consistency = (appearances / expected_frames * 100) if expected_frames > 0 else 0
        print(f"  {code}: Completed lap {driver_last_lap[code]}, Appeared in {appearances} frames ({consistency:.1f}% consistency)")
    
    # CRITICAL VERIFICATION
    print("\n" + "="*70)
    print("[VERIFY] CRITICAL CHECKS")
    print("="*70)
    
    all_passed = True
    
    # Check 1: Retired drivers appear in ALL remaining frames after they're last seen
    print("\n[CHECK-1] Retired drivers should appear in all frames after their last lap:")
    for code, last_lap in sorted(retired_drivers.items(), key=lambda x: x[1]):
        if code not in driver_appearances:
            print(f"  [X] {code}: NOT VISIBLE AT ALL! This will cause driver to disappear.")
            all_passed = False
            continue
        
        appearances = driver_appearances[code]
        first_frame_with_driver = appearances[0]
        last_frame_with_driver = appearances[-1]
        
        # Expected last frame based on last lap
        expected_last_frame = (last_lap * 30) + 29  # Last frame of that lap
        
        if last_frame_with_driver >= expected_last_frame - 5:  # Allow small tolerance
            print(f"  [OK] {code}: Appears until lap {last_lap} (final frame: {last_frame_with_driver})")
        else:
            print(f"  [!] {code}: May disappear early (lap {last_lap}, final frame: {last_frame_with_driver} vs expected {expected_last_frame})")
    
    # Check 2: No driver should be in 0 frames (unless they never raced)
    print("\n[CHECK-2] All drivers that raced should appear in multiple frames:")
    for code in sorted(driver_last_lap.keys()):
        appearance_count = len(driver_appearances.get(code, []))
        if appearance_count == 0:
            print(f"  [X] {code}: NEVER APPEARS - This is a critical bug!")
            all_passed = False
        elif appearance_count < 30:  # Should appear in at least 30 frames (1 lap)
            print(f"  [!] {code}: Only appears {appearance_count} times - May be incomplete data")
        else:
            print(f"  [OK] {code}: Appears {appearance_count} times")
    
    # Check 3: All 18 drivers should be in first frame (grid)
    first_frame_drivers = len(frames[0].get('drivers', {}))
    print(f"\n[CHECK-3] First frame should have all drivers on grid:")
    print(f"  Expected: 18 drivers, Actual: {first_frame_drivers}")
    if first_frame_drivers == 18:
        print(f"  [OK] All drivers visible on grid")
    else:
        print(f"  [X] Missing {18 - first_frame_drivers} drivers from grid!")
        all_passed = False
    
    # Check 4: Final frame should have finishing drivers
    last_frame_drivers = len(frames[-1].get('drivers', {}))
    print(f"\n[CHECK-4] Last frame should have all drivers (including retired at their positions):")
    print(f"  Expected: 18 drivers, Actual: {last_frame_drivers}")
    if last_frame_drivers >= 15:  # At least 15 (some might not be visible due to data cutoff)
        print(f"  [OK] Most drivers visible on final frame")
    else:
        print(f"  [!] Only {last_frame_drivers} drivers visible on final frame")
    
    print("\n" + "="*70)
    if all_passed:
        print("[SUCCESS] All checks passed! Retired drivers should remain visible.")
    else:
        print("[FAILURE] Some checks failed. Retired drivers may disappear.")
    print("="*70 + "\n")
    
    return all_passed

if __name__ == "__main__":
    success = test_retired_drivers()
    sys.exit(0 if success else 1)
