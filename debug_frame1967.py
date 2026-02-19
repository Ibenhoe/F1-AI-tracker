import json, os, tempfile

cache_file = os.path.join(tempfile.gettempdir(), 'fastf1_cache', 'race_2024_21_laps.json')
with open(cache_file) as f:
    laps = json.load(f)

for driver in ['STR', 'ALB']:
    driver_laps = [l for l in laps if l['driver'] == driver]
    print(f'{driver}: {len(driver_laps)} laps total, last lap = {driver_laps[-1]["lap_number"] if driver_laps else "none"}')
    for lap in driver_laps[-4:]:
        pts = lap.get('telemetry_points', [])
        valid = [p for p in pts if p.get('x') not in (None, 0) and p.get('y') not in (None, 0)]
        last_x = pts[-1].get('x') if pts else 'N/A'
        last_y = pts[-1].get('y') if pts else 'N/A'
        lap_num = lap['lap_number']
        print(f'  Lap {lap_num}: {len(pts)} pts, {len(valid)} valid, last=({last_x},{last_y})')

# Also check if ALB/STR appear in any lap 17 records
print()
lap17_drivers = [l['driver'] for l in laps if l['lap_number'] == 17]
print(f'Drivers with lap 17 data: {sorted(lap17_drivers)}')
print(f'STR in lap 17: {"STR" in lap17_drivers}')
print(f'ALB in lap 17: {"ALB" in lap17_drivers}')


nonzero = {code: (d['x'], d['y']) for code, d in drivers.items()
           if d.get('x') not in (None, 0) or d.get('y') not in (None, 0)}
zero = [code for code, d in drivers.items()
        if d.get('x') in (None, 0) and d.get('y') in (None, 0)]
null_xy = [code for code, d in drivers.items()
           if d.get('x') is None or d.get('y') is None]

print(f'  Non-zero x/y: {len(nonzero)} drivers -> {list(nonzero.keys())[:5]}')
print(f'  x=0,y=0:      {len(zero)} drivers -> {zero[:5]}')
print(f'  x/y is None:  {len(null_xy)} drivers -> {null_xy[:5]}')

# Scan ALL frames to find frames where <5 drivers have valid x/y
print('\nScanning all frames for drivers with x=0,y=0 issues...')
problem_laps = {}
for i, fr2 in enumerate(frames):
    d2 = fr2['drivers']
    bad = [code for code, d in d2.items()
           if d.get('x') in (None, 0) and d.get('y') in (None, 0)]
    if len(bad) > 5:
        lap2 = fr2.get('lap')
        if lap2 not in problem_laps:
            problem_laps[lap2] = {'count': 0, 'examples': bad[:5]}
        problem_laps[lap2]['count'] += 1

if problem_laps:
    print(f'Laps with >5 drivers at (0,0) in at least 1 frame:')
    for lap_num, info in sorted(problem_laps.items()):
        print(f'  Lap {lap_num}: {info["count"]} frames affected, drivers: {info["examples"]}')
else:
    print('No frames with >5 drivers at (0,0) found - fix works!')


# First peek at raw structure
frames = data['frames']
fr0 = frames[0]
print(f'Frame[0] type={type(fr0)}, keys={list(fr0.keys()) if isinstance(fr0, dict) else "n/a"}')
drivers0 = fr0.get('drivers', {})
print(f'drivers type={type(drivers0)}')
if isinstance(drivers0, dict):
    first_code = list(drivers0.keys())[0]
    print(f'First driver code={first_code}, value keys={list(drivers0[first_code].keys())}')
elif isinstance(drivers0, list) and drivers0:
    print(f'First driver keys={list(drivers0[0].keys())}')

import sys; sys.exit(0)


frames = data['frames']
total_laps = data.get('totalLaps', 1)
print(f'Total frames: {len(frames)}, Total laps: {total_laps}')
print(f'Frames per lap: ~{len(frames) // total_laps}')

# Check frame 1967
fr = frames[1967]
print(f'\nFrame 1967 (lap={fr.get("lap","?")}):')
print(f'  Total drivers in frame: {len(fr["drivers"])}')
for d in fr['drivers'][:8]:
    x, y = d.get('x'), d.get('y')
    print(f'  {d["driver"]}: x={x}, y={y}')

# Find frames where most drivers have x=0/y=0 or None (invisible)
print('\nScanning all frames for visibility issues...')
problem_frames = []
for i, frame in enumerate(frames):
    visible = [d for d in frame['drivers']
               if d.get('x') not in (None, 0) and d.get('y') not in (None, 0)]
    total = len(frame['drivers'])
    if total > 5 and len(visible) <= 1:
        problem_frames.append((i, len(visible), total, frame.get('lap')))

print(f'Frames with <=1 visible driver (but >5 total): {len(problem_frames)}')
for idx, vis, tot, lap in problem_frames[:20]:
    print(f'  Frame {idx}: {vis}/{tot} visible, lap={lap}')

# Also check what happens between frames 1960-1975 specifically
print('\nFrames 1960-1975 detail:')
for i in range(1960, min(1976, len(frames))):
    fr2 = frames[i]
    visible = [d["driver"] for d in fr2['drivers']
               if d.get('x') not in (None, 0) and d.get('y') not in (None, 0)]
    print(f'  Frame {i} (lap={fr2.get("lap","?")}): {len(visible)}/{len(fr2["drivers"])} visible -> {visible[:4]}')
