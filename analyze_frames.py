#!/usr/bin/env python3
"""
Debug script to identify frames with no position changes
"""
import json
import sys

def analyze_frames():
    """Analyze which frames have no action"""
    try:
        cache_file = 'cache/race_21_frames.json'
        
        print("\n" + "="*80)
        print("[ANALYSIS] Checking for static frames...")
        print("="*80)
        
        with open(cache_file, 'r') as f:
            data = json.load(f)
        
        frames = data.get('frames', [])
        print(f"\nTotal frames: {len(frames)}")
        
        # Find frames where nothing changes
        static_frame_ranges = []
        static_start = None
        prev_positions = None
        
        for frame_idx, frame in enumerate(frames):
            drivers = frame.get('drivers', {})
            
            if not drivers:
                if static_start is None:
                    static_start = frame_idx
                continue
            
            current_positions = {code: driver.get('position') for code, driver in drivers.items()}
            
            # Check if positions changed from previous frame
            if prev_positions and current_positions == prev_positions:
                if static_start is None:
                    static_start = frame_idx
            else:
                if static_start is not None:
                    static_frame_ranges.append((static_start, frame_idx - 1))
                    static_start = None
            
            prev_positions = current_positions
        
        if static_start is not None:
            static_frame_ranges.append((static_start, len(frames) - 1))
        
        print(f"\nStatic frame ranges (no position changes):")
        for start, end in static_frame_ranges:
            count = end - start + 1
            print(f"  Frames {start:5d}-{end:5d}: {count:4d} frames ({(count/len(frames)*100):.1f}%)")
        
        total_static = sum(end - start + 1 for start, end in static_frame_ranges)
        print(f"\nTotal static frames: {total_static} ({(total_static/len(frames)*100):.1f}%)")
        
        # Check first few frames
        print(f"\n[FIRST FRAMES] Checking first 10 frames:")
        for i in range(min(10, len(frames))):
            frame = frames[i]
            drivers = frame.get('drivers', {})
            lap = frame.get('lap', '?')
            print(f"  Frame {i:3d}, Lap {lap}: {len(drivers)} drivers")
            if drivers:
                positions = [d.get('position') for d in drivers.values()]
                print(f"    Positions: {[f'{p:.1f}' for p in positions[:5]]}...")
        
        return len(static_frame_ranges) > 0
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == '__main__':
    has_static = analyze_frames()
    sys.exit(0)
