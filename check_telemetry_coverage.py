#!/usr/bin/env python3
"""Check telemetry coverage per driver"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fastf1_data_fetcher import FastF1DataFetcher

fetcher = FastF1DataFetcher()
fetcher.fetch_race(year=2024, round_number=21)
laps_data = fetcher.process_race_laps_streaming(show_progress=False)

# Group by driver
drivers = {}
for lap in laps_data:
    code = lap['driver']
    if code not in drivers:
        drivers[code] = {'total': 0, 'with_xy': 0, 'x_null': 0}
    
    drivers[code]['total'] += 1
    if lap.get('x') is not None and lap.get('y') is not None:
        drivers[code]['with_xy'] += 1
    else:
        drivers[code]['x_null'] += 1

print("Driver | Total Laps | With X,Y | X,Y% | Null X,Y | Notes")
print("-------|-----------|----------|------|----------|-------")
for code in sorted(drivers.keys()):
    d = drivers[code]
    pct = 100 * d['with_xy'] // d['total']
    status = "[OK]" if pct == 100 else "[MISSING]"
    print(f"{code:6} | {d['total']:9} | {d['with_xy']:8} | {pct:3}% | {d['x_null']:8} | {status}")

# Check first 3 and last 3 laps for a driver with data
print("\n\nNOR first 3 laps:")
nor_laps = [l for l in laps_data if l['driver'] == 'NOR'][:3]
for lap in nor_laps:
    x = lap.get('x')
    y = lap.get('y')
    has_xy = "OK" if (x is not None and y is not None) else "BAD"
    print(f"  Lap {int(lap['lap_number']):3}: {has_xy} x={x is not None}, y={y is not None}")
