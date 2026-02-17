import sys
sys.path.insert(0, '.')
from fastf1_data_fetcher import FastF1DataFetcher

fetcher = FastF1DataFetcher()
fetcher.fetch_race(year=2024, round_number=21)
laps = fetcher.process_race_laps_streaming(show_progress=False)

drivers = set(l['driver'] for l in laps)
print(f'Total drivers: {len(drivers)}')
print('Drivers:', sorted(drivers))

alb = [l for l in laps if l['driver'] == 'ALB']
print(f'\nALB has {len(alb)} laps')
if alb:
    print(f'ALB lap 1: x={alb[0].get("x")}, y={alb[0].get("y")}')
    
# Count how many drivers have data
for driver_code in sorted(drivers):
    driver_laps = [l for l in laps if l['driver'] == driver_code]
    has_xy = sum(1 for l in driver_laps if l.get('x') and l.get('y'))
    print(f'{driver_code}: {has_xy}/{len(driver_laps)} laps with x,y')
