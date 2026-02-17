#!/usr/bin/env python3
import fastf1
import pandas as pd
import os
import tempfile

# Enable cache
cache_dir = os.path.join(tempfile.gettempdir(), 'fastf1_cache')
fastf1.Cache.enable_cache(cache_dir)

from app import _extract_track_from_telemetry

try:
    print("[TEST] Loading Brazil 2024 race...")
    session = fastf1.get_session(2024, 21, 'R')
    session.load(telemetry=True, weather=False)
    
    print("[TEST] Extracting track coordinates...")
    track = _extract_track_from_telemetry(session)
    
    if track:
        print(f"[SUCCESS] Got {len(track)} track points!")
        print(f"[SAMPLE] First 5 points:")
        for i, point in enumerate(track[:5]):
            print(f"  Point {i}: x={point['x']:.1f}, y={point['y']:.1f}")
        
        # Check bounds
        xs = [p['x'] for p in track]
        ys = [p['y'] for p in track]
        print(f"\n[BOUNDS]")
        print(f"  X range: {min(xs):.1f} to {max(xs):.1f}")
        print(f"  Y range: {min(ys):.1f} to {max(ys):.1f}")
    else:
        print("[FAILED] No track extracted")
        
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()
