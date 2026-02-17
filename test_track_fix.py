#!/usr/bin/env python3
"""Quick test of track extraction fix"""

import sys
sys.path.insert(0, '.')

from fastf1_data_fetcher import FastF1DataFetcher
import fastf1

print("[TEST] Loading race 21...")
session = fastf1.get_session(2024, 21, 'R')
session.load(telemetry=True, weather=False)

print("[TEST] Session loaded, extracting track...")

# Get laps
all_laps = session.laps
print(f"[TEST] Session has {len(all_laps)} lap records")

# Extract coordinates like the fixed function does
coords_set = set()
coords_list = []

sample_laps = min(10, len(all_laps))
print(f"[TEST] Sampling first {sample_laps} laps...")

for lap_idx in range(sample_laps):
    try:
        lap_index = all_laps.index[lap_idx]
        actual_lap = all_laps.loc[lap_index]
        
        if not hasattr(actual_lap, 'get_telemetry'):
            print(f"  Lap {lap_idx}: No get_telemetry method")
            continue
        
        telemetry = actual_lap.get_telemetry()
        if telemetry is None or len(telemetry) == 0:
            print(f"  Lap {lap_idx}: No telemetry data")
            continue
        
        print(f"  Lap {lap_idx}: {len(telemetry)} telemetry points")
        
        if 'X' in telemetry.columns and 'Y' in telemetry.columns:
            step = max(1, len(telemetry) // 100)
            for idx in range(0, len(telemetry), step):
                row = telemetry.iloc[idx]
                try:
                    x = float(row['X'])
                    y = float(row['Y'])
                    coord_tuple = (round(x, 1), round(y, 1))
                    if coord_tuple not in coords_set:
                        coords_set.add(coord_tuple)
                        coords_list.append({'x': x, 'y': y})
                except:
                    pass
    except Exception as e:
        print(f"  Lap {lap_idx}: Error - {e}")

print(f"\n[RESULT] Extracted {len(coords_list)} unique coordinates")

if coords_list:
    xs = [p['x'] for p in coords_list]
    ys = [p['y'] for p in coords_list]
    
    bounds = {
        'minX': min(xs),
        'maxX': max(xs),
        'minY': min(ys),
        'maxY': max(ys),
    }
    
    print(f"\n[BOUNDS]")
    print(f"  X: {bounds['minX']:.1f} to {bounds['maxX']:.1f} (range: {bounds['maxX']-bounds['minX']:.1f})")
    print(f"  Y: {bounds['minY']:.1f} to {bounds['maxY']:.1f} (range: {bounds['maxY']-bounds['minY']:.1f})")
    
    print(f"\n[SAMPLE] First 5 coordinates:")
    for i, coord in enumerate(coords_list[:5]):
        print(f"  {i+1}. x={coord['x']:.1f}, y={coord['y']:.1f}")
else:
    print("[ERROR] No coordinates extracted - track extraction failing!")
