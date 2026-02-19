#!/usr/bin/env python3
"""Test RaceSimulator with real FastF1 data to verify gaps are realistic"""

import sys
sys.path.insert(0, '.')

import fastf1
import os
import tempfile
from race_simulator import RaceSimulator
from continuous_model_learner_v2 import ContinuousModelLearner

# Setup FastF1 cache
cache_dir = os.path.join(tempfile.gettempdir(), 'fastf1_cache')
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

print("[TEST] Loading Race 1 (Bahrain 2024)...")
session = fastf1.get_session(2024, 1, 'R')
session.load()

# Get drivers from qualifying
print("[TEST] Getting drivers from qualifying...")
drivers = []
try:
    qual = fastf1.get_session(2024, 1, 'Q')
    qual.load(telemetry=False, weather=False)
    
    for grid_idx, (_, qual_row) in enumerate(qual.results.iterrows(), 1):
        driver_code = str(qual_row.get('Abbreviation', ''))
        if driver_code and driver_code != 'nan':
            drivers.append({
                'code': driver_code,
                'name': str(qual_row.get('FullName', 'Unknown')),
                'team': str(qual_row.get('TeamName', 'Unknown')),
                'number': int(qual_row.get('DriverNumber', 0)),
                'grid_position': grid_idx
            })
except Exception as e:
    print(f"[TEST] Could not load qualifying: {e}")

print(f"[TEST] ✓ Loaded {len(drivers)} drivers")

# Initialize model
print("[TEST] Initializing AI model...")
model = ContinuousModelLearner()

# Initialize race simulator
print("[TEST] Initializing RaceSimulator...")
sim = RaceSimulator(
    race_number=1,
    model=model,
    laps_data=session.laps,
    drivers=drivers,
    weather_data=None
)

# Simulate first 5 laps
print("\n[TEST] Simulating first 5 laps...")
for lap in range(1, 6):
    lap_state = sim.simulate_lap(lap)
    
    print(f"\n[LAP {lap}] Overview:")
    top_3 = lap_state['drivers'][:3]
    for i, driver in enumerate(top_3, 1):
        gap = driver.get('gap_to_leader', 0)
        gap_next = driver.get('gap_to_next', 0)
        print(f"  P{i}: {driver.get('driver', '?'):3s} - gap_to_leader={gap:6.2f}s gap_to_next={gap_next:6.2f}s")

print("\n[TEST] ✓ Complete! Check if gaps look realistic.")
