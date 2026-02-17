#!/usr/bin/env python3
"""Simple telemetry test"""

from fastf1_data_fetcher import FastF1DataFetcher
import json

fetcher = FastF1DataFetcher()
fetcher.fetch_race(year=2024, round_number=21)
laps = fetcher.process_race_laps_streaming(show_progress=False)

# Count and show results
tel_count = sum(1 for lap in laps if lap.get('x') is not None)
pass_rate = 100 * tel_count // len(laps) if laps else 0

results = {
    'total_laps': len(laps),
    'telemetry_count': tel_count,
    'pass_rate': f"{pass_rate}%",
    'sample_laps': []
}

# Add first 5 laps with telemetry
for lap in laps[:5]:
    if lap.get('x') is not None:
        results['sample_laps'].append({
            'driver': lap['driver'],
            'lap': int(lap['lap_number']),
            'x': round(lap['x'], 2),
            'y': round(lap['y'], 2),
            'speed': lap['speed'],
            'gear': lap['gear']
        })
        break

# Print results
print(f"Total: {results['total_laps']} laps")
print(f"Telemetry: {results['telemetry_count']}/{results['total_laps']} ({results['pass_rate']})")
if results['sample_laps']:
    s = results['sample_laps'][0]
    print(f"Sample: {s['driver']} L{s['lap']} -> x={s['x']}, y={s['y']}")
else:
    print("ERROR: No telemetry extracted!")

# Also write to file
with open('test_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nResults saved to test_results.json")
