#!/usr/bin/env python3
"""
Test race simulation directly - bypass frontend
"""

import time
from race_simulator import RaceSimulator
from continuous_model_learner_v2 import ContinuousModelLearner
from fastf1_data_fetcher import FastF1DataFetcher

print("\n" + "="*70)
print("[TEST] Direct Race Simulation Test")
print("="*70 + "\n")

# Fetch race data
print("[FETCH] Loading FastF1 data for race 21...")
fetcher = FastF1DataFetcher()
fetcher.fetch_race(2024, 21)
drivers = []

try:
    import fastf1
    qual_session = fastf1.get_session(2024, 21, 'Q')
    qual_session.load()
    
    for grid_idx, (_, qual_row) in enumerate(qual_session.results.iterrows()):
        driver_code = str(qual_row.get('Abbreviation', ''))
        if driver_code and driver_code != 'nan':
            drivers.append({
                'code': driver_code,
                'name': str(qual_row.get('FullName', 'Unknown')),
                'team': str(qual_row.get('TeamName', 'Unknown')),
                'number': int(qual_row.get('DriverNumber', 0)),
                'grid_position': grid_idx + 1
            })
    
    print(f"[OK] Loaded {len(drivers)} drivers from qualifying")
    
except Exception as e:
    print(f"[ERROR] Could not load qualifying: {e}")
    drivers = []

# Load lap data
laps = fetcher.session.laps if hasattr(fetcher.session, 'laps') else None
weather_data = fetcher.session.weather_data if hasattr(fetcher.session, 'weather_data') else None

print(f"[OK] Lap data: {len(laps) if laps is not None else 0} rows")
if laps is not None:
    print(f"[INFO] Lap columns: {laps.columns.tolist()[:10]}")

# Initialize model
print("\n[MODEL] Initializing AI model...")
model = ContinuousModelLearner()
model.pretrain_on_historical_data('f1_historical_5years.csv')
print("[OK] Model pre-trained")

# Create simulator
print("\n[SIMULATOR] Creating RaceSimulator...")
simulator = RaceSimulator(
    race_number=21,
    model=model,
    laps_data=laps,
    drivers=drivers,
    weather_data=weather_data
)
print("[OK] RaceSimulator created")

# Simulate first 5 laps
print("\n" + "="*70)
print("[RUN] Simulating laps 1-5")
print("="*70 + "\n")

for lap_num in range(1, 6):
    print(f"\n[LAP {lap_num}] Simulating...")
    lap_state = simulator.simulate_lap(lap_num)
    
    print(f"  Drivers: {len(lap_state['drivers'])}")
    print(f"  Events: {len(lap_state['events'])}")
    
    # Show top 5 drivers
    print(f"\n  [TOP 5 POSITIONS]:")
    for driver in lap_state['drivers'][:5]:
        gap_to_leader = driver.get('gap_to_leader', 0.0)
        gap_to_next = driver.get('gap_to_next', 0.0)
        print(f"    P{driver.get('position'):.0f}: {driver.get('driver'):3s} - gap_to_leader={gap_to_leader:.2f}s, gap_to_next={gap_to_next:.2f}s")
    
    # Show events
    if lap_state['events']:
        print(f"\n  [EVENTS] {len(lap_state['events'])} event(s):")
        for event in lap_state['events']:
            print(f"    - {event.get('type')}/{event.get('subtype')}: {event.get('message')}")
    else:
        print(f"\n  [EVENTS] No events this lap")

print("\n" + "="*70)
print("[DONE] Simulation complete!")
print("="*70 + "\n")
