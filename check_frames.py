import json
try:
    with open('cache/race_21_frames.json') as f:
        d = json.load(f)
    frames = d.get('frames', [])
    print(f"Frames: {len(frames)}")
    if frames:
        f = frames[min(100, len(frames)-1)]
        drivers = f.get('drivers', {})
        print(f"Frame {f['frameIndex']}: lap {f['lap']}, drivers={len(drivers)}")
        
        # Check if drivers have telemetry points
        ver = drivers.get('VER')
        if ver:
            print(f"VER: x={ver.get('x')}, y={ver.get('y')}, speed={ver.get('speed')}")
except Exception as e:
    print(f"Error: {e}")
    import os
    print(f"Cache exists: {os.path.exists('cache/race_21_frames.json')}")
