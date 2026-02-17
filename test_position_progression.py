#!/usr/bin/env python3
"""
Test to check if driver positions are monotonically increasing (not going backwards)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastf1_data_fetcher import FastF1DataFetcher
from app import _build_replay_frames

def test_position_progression():
    """Check if drivers move forward (increase position) during race"""
    
    print("\n" + "="*70)
    print("[TEST] Driver Position Progression Check")
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
    
    if not frames or len(frames) < 60:
        print("ERROR: Not enough frames generated")
        return False
    
    # Track position changes for a few drivers
    test_drivers = ['VER', 'RUS', 'NOR', 'HAM']
    
    print("\n[TRACKING] Position changes for selected drivers across frames:")
    print("-" * 70)
    
    all_issues = []
    
    for driver_code in test_drivers:
        positions = []
        frame_indices = []
        
        # Sample frames across the race
        sample_frames = [0, 60, 300, 600, 1035, 1500, 2069]  # Start, early, mid, late, final
        
        for frame_idx in sample_frames:
            if frame_idx < len(frames):
                frame = frames[frame_idx]
                if driver_code in frame.get('drivers', {}):
                    pos = frame['drivers'][driver_code].get('position')
                    if pos is not None:
                        positions.append(pos)
                        frame_indices.append(frame_idx)
        
        # Print position changes
        print(f"\n{driver_code} (Frames: {frame_indices[0]} -> {frame_indices[-1]}):")
        for fi, pos in zip(frame_indices, positions):
            lap_num = fi // 30 + 1
            print(f"  Frame {fi:4d} (Lap ~{lap_num:2d}): Position {pos:5.1f}")
        
        # Check for backward movement
        for i in range(1, len(positions)):
            if positions[i] > positions[i-1] + 0.5:  # Allow small tolerance
                all_issues.append(f"{driver_code}: Goes backward from {positions[i-1]:.1f} to {positions[i]:.1f}")
                print(f"    WARNING: Position increased (backward movement?) from {positions[i-1]:.1f} to {positions[i]:.1f}")
    
    print("\n" + "="*70)
    if all_issues:
        print(f"[ISSUES FOUND] {len(all_issues)} position anomalies detected:")
        for issue in all_issues:
            print(f"  - {issue}")
        return False
    else:
        print("[OK] All drivers show correct forward/neutral position progression")
        return True

if __name__ == "__main__":
    success = test_position_progression()
    sys.exit(0 if success else 1)
