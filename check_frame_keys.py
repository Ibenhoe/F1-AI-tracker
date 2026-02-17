import json

f = json.load(open('cache/race_21_frames.json'))
frame = f['frames'][0]
ver = frame['drivers'].get('VER')

print(f"VER keys: {list(ver.keys())}")
print(f"Has telemetry_points: {'telemetry_points' in ver}")
print(f"x = {ver.get('x')}")
print(f"y = {ver.get('y')}")
