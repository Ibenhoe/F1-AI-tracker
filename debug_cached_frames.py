#!/usr/bin/env python3
"""Inspect cached animation frames"""

import json
import os
from pathlib import Path

cache_dir = Path(os.path.expandvars(r'%TEMP%\fastf1_cache'))
frames_file = cache_dir / 'race_2024_21_frames.json'

print(f"[CHECK] Looking for cached frames at: {frames_file}")

if not frames_file.exists():
    print(f"  ❌ File NOT found. Cache may need regeneration.")
    print(f"\n[INFO] Available cache files:")
    for f in cache_dir.glob('race_2024_21*'):
        print(f"  - {f.name}")
    exit(1)

print(f"  ✅ File found, loading...")

with open(frames_file, 'r', encoding='utf-8') as f:
    frames = json.load(f)

print(f"\n[FRAMES] Total frames: {len(frames)}")

# Find lap 3 frames  
lap3_frames = [f for f in frames if f['lap'] == 3]
print(f"[FRAMES] Lap 3 has {len(lap3_frames)} frames")

if lap3_frames:
    # Check first frame
    first_frame = lap3_frames[0]
    print(f"\n[SAMPLE] First frame of lap 3:")
    print(f"  frameIndex: {first_frame['frameIndex']}")
    print(f"  lap: {first_frame['lap']}")
    
    # Check one driver
    if 'VER' in first_frame['drivers']:
        ver = first_frame['drivers']['VER']
        print(f"  Driver VER:")
        print(f"    position: {ver.get('position')}")
        print(f"    x: {ver.get('x')}")
        print(f"    y: {ver.get('y')}")
        print(f"    speed: {ver.get('speed')}")
        print(f"    gear: {ver.get('gear')}")
        print(f"    throttle: {ver.get('throttle')}")
        print(f"    brake: {ver.get('brake')}")
    
    # Check multiple frames in lap 3 to see if x,y changes
    print(f"\n[CHECK] Do drivers move within lap 3? (checking x,y across frames)")
    ver_x_values = [f['drivers']['VER']['x'] for f in lap3_frames if 'VER' in f['drivers']]
    ver_y_values = [f['drivers']['VER']['y'] for f in lap3_frames if 'VER' in f['drivers']]
    
    print(f"  VER x range: {min(ver_x_values):.1f} to {max(ver_x_values):.1f} (movement: {max(ver_x_values) - min(ver_x_values):.1f})")
    print(f"  VER y range: {min(ver_y_values):.1f} to {max(ver_y_values):.1f} (movement: {max(ver_y_values) - min(ver_y_values):.1f})")
    
    # Check if all frames have same values (interpolation not working)
    unique_x = len(set(round(v, 1) for v in ver_x_values))
    unique_y = len(set(round(v, 1) for v in ver_y_values))
    print(f"  Unique x values: {unique_x} / {len(ver_x_values)} frames")
    print(f"    -> {'✅ Interpolation working' if unique_x > 1 else '❌ All same, no interpolation'}")
    print(f"  Unique y values: {unique_y} / {len(ver_y_values)} frames")
    print(f"    -> {'✅ Interpolation working' if unique_y > 1 else '❌ All same, no interpolation'}")

print(f"\n[DONE]")
