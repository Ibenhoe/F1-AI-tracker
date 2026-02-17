#!/usr/bin/env python3
"""
Quick test script to validate frame generation with position interpolation fix
"""

import json
import os
from app import _build_replay_frames, _fetch_fastf1_data
from fastf1_data_fetcher import FastF1DataFetcher

def test_frame_generation():
    """Test frame generation with position interpolation"""
    
    race_num = 21  # Abu Dhabi
    print("[TEST] Testing frame generation for race 21 (Abu Dhabi)")
    
    # Fetch real race data
    fetcher = FastF1DataFetcher()
    if not fetcher.fetch_race(2024, race_num):
        print("ERROR: Could not fetch race data")
        return False
    
    race_info = fetcher.get_race_summary()
    laps_data = fetcher.process_race_laps_streaming()
    
    print(f"[TEST] Loaded {len(laps_data)} lap records")
    
    # Generate frames
    print("[TEST] Generating frames...")
    frames = _build_replay_frames(laps_data, race_info)
    
    print(f"[TEST] Generated {len(frames)} frames")
    
    # Analyze frames
    if len(frames) > 0:
        # Check first frame (lap 1)
        first_frame = frames[0]
        print(f"\n[FIRST FRAME] Lap {first_frame['lap']}")
        print(f"  Drivers: {len(first_frame['drivers'])}")
        
        drivers_sorted = sorted(first_frame['drivers'].values(), key=lambda d: d['position'])
        for i, driver in enumerate(drivers_sorted[:5]):
            print(f"    {i+1}. {driver['code']}: P{driver['position']:.1f}")
        
        # Check middle frame (around lap 30)
        mid_idx = len(frames) // 2
        mid_frame = frames[mid_idx]
        print(f"\n[MIDDLE FRAME] Lap {mid_frame['lap']}, Frame {mid_idx}")
        print(f"  Drivers: {len(mid_frame['drivers'])}")
        
        drivers_sorted = sorted(mid_frame['drivers'].values(), key=lambda d: d['position'])
        for i, driver in enumerate(drivers_sorted[:5]):
            print(f"    {i+1}. {driver['code']}: P{driver['position']:.1f}")
        
        # Check last frame
        last_frame = frames[-1]
        print(f"\n[LAST FRAME] Lap {last_frame['lap']}")
        print(f"  Drivers: {len(last_frame['drivers'])}")
        
        drivers_sorted = sorted(last_frame['drivers'].values(), key=lambda d: d['position'])
        for i, driver in enumerate(drivers_sorted[:5]):
            print(f"    {i+1}. {driver['code']}: P{driver['position']:.1f}")
        
        # Check for position anomalies
        print("\n[ANOMALY CHECK]")
        anomalies = 0
        for frame_idx, frame in enumerate(frames[1:], 1):
            prev_frame = frames[frame_idx - 1]
            
            for code in frame['drivers']:
                if code in prev_frame['drivers']:
                    prev_pos = prev_frame['drivers'][code]['position']
                    curr_pos = frame['drivers'][code]['position']
                    
                    # Check if position goes backwards (negative delta larger than 1.5)
                    if curr_pos < prev_pos - 1.5:
                        anomalies += 1
                        if anomalies <= 5:  # Show first 5 only
                            print(f"  Frame {frame_idx}: {code} jumped from P{prev_pos:.1f} to P{curr_pos:.1f} (delta: {curr_pos - prev_pos:.2f})")
        
        if anomalies == 0:
            print("  ✓ No position anomalies detected!")
        else:
            print(f"  ⚠ Found {anomalies} position anomalies")
        
        print("\n[TEST] ✓ Frame generation test complete!")
        return True
    else:
        print("[TEST] ERROR: No frames generated")
        return False

if __name__ == "__main__":
    success = test_frame_generation()
    exit(0 if success else 1)
