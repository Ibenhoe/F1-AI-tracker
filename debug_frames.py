#!/usr/bin/env python3
"""Debug script to inspect lap data"""

import json
import sys
from pathlib import Path

# Add current dir to path
sys.path.insert(0, str(Path(__file__).parent))

from fastf1_data_fetcher import FastF1DataFetcher

print("[DEBUG] Loading race 21 data...\n")

# Load race data
fetcher = FastF1DataFetcher()
fetcher.fetch_race(year=2024, round_number=21)
laps_data = fetcher.process_race_laps_streaming(show_progress=False)

print(f"[DEBUG] Loaded {len(laps_data)} laps")

print(f"\n[SAMPLE] First 5 laps data (all drivers):")
print("Lap | Driver | Position | X         | Y         | Speed")
print("----|--------|----------|-----------|-----------|--------")
for lap in laps_data[:5]:
    x = lap.get('x')
    y = lap.get('y')
    x_str = f"{x:.1f}" if x is not None else "None"
    y_str = f"{y:.1f}" if y is not None else "None"
    speed = lap.get('speed')
    speed_str = f"{speed:.1f}" if speed is not None else "None"
    print(f"{int(lap['lap_number']):3d} | {lap['driver']:6s} | {int(lap['position']):8.0f} | {x_str:9s} | {y_str:9s} | {speed_str}")

print(f"\n[CHECK] How many laps have x,y telemetry?")
with_xy = sum(1 for l in laps_data if l.get('x') is not None and l.get('y') is not None)
print(f"  {with_xy}/{len(laps_data)} laps have x,y coordinates ({100*with_xy//len(laps_data)}%)")

print(f"\n[CHECK] VER's lap 1-5 positions and coordinates:")
ver_laps = [l for l in laps_data if l['driver'] == 'VER'][:5]
for lap in ver_laps:
    x = lap.get('x')
    y = lap.get('y')
    print(f"  Lap {int(lap['lap_number'])}: pos={lap['position']:2.0f}, x={x}, y={y}")

print(f"\n[CHECK] NOR's lap 1-5 positions and coordinates:")
nor_laps = [l for l in laps_data if l['driver'] == 'NOR'][:5]
for lap in nor_laps:
    x = lap.get('x')
    y = lap.get('y')
    print(f"  Lap {int(lap['lap_number'])}: pos={lap['position']:2.0f}, x={x}, y={y}")

print(f"\n[DONE]")
