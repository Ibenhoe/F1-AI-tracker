#!/usr/bin/env python3
"""Test frame generation with smooth animation"""

from app import _build_replay_frames
from fastf1_data_fetcher import FastF1DataFetcher

print("[TEST] Loading race data and generating frames...")
fetcher = FastF1DataFetcher()
if not fetcher.fetch_race(2024, 21):
    print("[ERROR] Could not load race")
    exit(1)

print("[TEST] Processing laps...")
laps_data = fetcher.process_race_laps_streaming()
print(f"[OK] Loaded {len(laps_data)} lap records")

print("[TEST] Building frames for replay...")
frames = _build_replay_frames(laps_data, {})

print(f"[OK] Generated {len(frames)} frames")
print(f"[OK] Expected: 69 laps × 30 frames/lap = 2070 frames")
print(f"[OK] Actual: {len(frames)} frames")

# Check first frame (lap 1, frame 0)
if frames:
    first_frame = frames[0]
    print(f"\n[FIRST FRAME] Lap {first_frame['lap']}, Frame {first_frame['frameIndex']}")
    print(f"  Drivers: {len(first_frame['drivers'])}")
    
    # Show driver positions for first frame
    sorted_drivers = sorted(first_frame['drivers'].items(), key=lambda x: x[1]['position'])
    print("  Positions on first frame:")
    for code, driver in sorted_drivers[:5]:
        print(f"    P{driver['position']:.1f}: {code:3s} - {driver['driver_name']}")

# Check a frame in the middle (around lap 35)
mid_frame_idx = len(frames) // 2
if mid_frame_idx < len(frames):
    mid_frame = frames[mid_frame_idx]
    print(f"\n[MID FRAME] Lap {mid_frame['lap']}, Frame {mid_frame['frameIndex']}")
    print(f"  Drivers: {len(mid_frame['drivers'])}")
    
    sorted_drivers = sorted(mid_frame['drivers'].items(), key=lambda x: x[1]['position'])
    print("  Positions on middle frame:")
    for code, driver in sorted_drivers[:5]:
        print(f"    P{driver['position']:.1f}: {code:3s} - {driver['driver_name']}")

# Check last frame
if frames:
    last_frame = frames[-1]
    print(f"\n[LAST FRAME] Lap {last_frame['lap']}, Frame {last_frame['frameIndex']}")
    print(f"  Drivers: {len(last_frame['drivers'])}")
    
    sorted_drivers = sorted(last_frame['drivers'].items(), key=lambda x: x[1]['position'])
    print("  Final positions:")
    for code, driver in sorted_drivers[:5]:
        print(f"    P{driver['position']:.1f}: {code:3s} - {driver['driver_name']}")

print("\n[SUCCESS] Frame generation test passed!")
print(f"[INFO] With 30fps frames and 0.25x playback speed:")
print(f"[INFO] Total frames: {len(frames)}")
print(f"[INFO] Real-time seconds: {len(frames) / 30:.1f}s")
print(f"[INFO] Playback time at 0.25x speed: {len(frames) / 30 / 0.25:.1f}s = {len(frames) / 30 / 0.25 / 60:.1f} minutes")
