import json

# Check what's actually in lap data
with open('c:\\Users\\ibenv\\AppData\\Local\\Temp\\fastf1_cache\\race_2024_21_laps.json') as f:
    laps = json.load(f)

print(f"Total laps: {len(laps)}\n")

# Check first lap for each driver
drivers = {}
for lap in laps:
    code = lap['driver']
    if code not in drivers:
        drivers[code] = lap

print("Driver | Lap 1 | Telemetry Points")
print("-------|-------|----------------")
for code in sorted(drivers.keys()):
    lap = drivers[code]
    points = lap.get('telemetry_points', [])
    print(f"{code:6} | {int(lap['lap_number']):3} | {len(points):6}")

# Check a driver with data
ver_laps = [l for l in laps if l['driver'] == 'VER']
print(f"\nVER first 3 laps:")
for lap in ver_laps[:3]:
    pts = lap.get('telemetry_points', [])
    if pts:
        first = pts[0]
        last = pts[-1]
        print(f"  Lap {int(lap['lap_number'])}: {len(pts)} points, x range: {first['x']:.0f} to {last['x']:.0f}")
    else:
        print(f"  Lap {int(lap['lap_number'])}: NO TELEMETRY!")
