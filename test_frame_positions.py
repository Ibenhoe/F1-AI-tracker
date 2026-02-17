#!/usr/bin/env python3
"""
Test script to verify frame positions are animating correctly
Checks that drivers move through frames and positions change smoothly
"""

import json
import requests
import sys

def test_frame_positions():
    """Test that frame positions change throughout replay"""
    try:
        print("\n" + "="*80)
        print("[TEST] Verifying Frame Position Animation")
        print("="*80)
        
        # Fetch replay data
        print("\n[1/3] Fetching replay data...")
        response = requests.get('http://localhost:5000/api/race/replay-data?race=21')
        
        if response.status_code != 200:
            print(f"  ERROR: Failed to fetch data (HTTP {response.status_code})")
            return False
        
        data = response.json()
        frames = data.get('frames', [])
        
        if not frames:
            print("  ERROR: No frames in response")
            return False
        
        print(f"  ✓ Loaded {len(frames)} frames")
        
        # Check frame positions for sample drivers
        print("\n[2/3] Analyzing position changes...")
        
        test_drivers = ['VER', 'HAM', 'LEC', 'NOR']  # Sample drivers
        
        for driver_code in test_drivers:
            print(f"\n  Driver {driver_code}:")
            
            # Get positions at different frame indices
            frame_samples = [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, len(frames)-1]
            positions = []
            
            for frame_idx in frame_samples:
                if frame_idx < len(frames):
                    frame = frames[frame_idx]
                    drivers_in_frame = frame.get('drivers', {})
                    
                    if driver_code in drivers_in_frame:
                        pos = drivers_in_frame[driver_code].get('position')
                        positions.append((frame_idx, pos))
            
            if positions:
                print(f"    Frame positions sampled:")
                for frame_idx, pos in positions:
                    frame_pct = (frame_idx / len(frames)) * 100
                    print(f"      Frame {frame_idx:5d} ({frame_pct:5.1f}%): Position {pos:.2f}")
                
                # Check for position changes
                pos_values = [p[1] for p in positions]
                pos_range = max(pos_values) - min(pos_values)
                
                if pos_range > 0.5:
                    print(f"    ✓ Position changed by {pos_range:.2f} positions (GOOD)")
                else:
                    print(f"    ⚠️  Position barely changed ({pos_range:.2f}) - may be standing still")
            else:
                print(f"    ERROR: Driver not found in frames")
        
        # Verify first vs last frame for all drivers
        print("\n[3/3] Checking grid vs final positions...")
        
        first_frame = frames[0].get('drivers', {})
        last_frame = frames[-1].get('drivers', {})
        
        position_changes = 0
        stationary_drivers = 0
        
        for driver_code in first_frame.keys():
            if driver_code in last_frame:
                first_pos = first_frame[driver_code].get('position')
                last_pos = last_frame[driver_code].get('position')
                
                change = abs(last_pos - first_pos)
                if change > 0.5:
                    position_changes += 1
                else:
                    stationary_drivers += 1
        
        total_drivers = len(first_frame)
        print(f"\n  Total drivers: {total_drivers}")
        print(f"  Position changed: {position_changes}")
        print(f"  Stationary: {stationary_drivers}")
        
        if position_changes > total_drivers * 0.5:
            print(f"  ✓ Most drivers moved through race (GOOD)")
            return True
        else:
            print(f"  ⚠️  Many drivers appear stationary (BAD)")
            return False
        
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_frame_positions()
    sys.exit(0 if success else 1)
