#!/usr/bin/env python3
"""Test improved pace-weighted scoring"""

from continuous_model_learner_v2 import ContinuousModelLearner
from fastf1_data_fetcher import FastF1DataFetcher

print("[TEST] Pace-Weighted Scoring - Driver 1 vs Driver 14")
print("="*70)

# Fetch Race 5 data
fetcher = FastF1DataFetcher()
if not fetcher.fetch_race(2024, 5):
    print("❌ Failed to fetch")
    exit(1)

# Get first few laps
import sys
sys.path.insert(0, '/content/workspace')

# Simulate pace difference scenario
# Driver 1 (P1): consistently 0.3s faster
# Driver 14 (P2): consistently slower

test_laps = [
    {
        'driver': 'Driver 1',
        'position': 1,
        'current_lap': 4,
        'lap_time': 95.5,
        'tire_compound': 'SOFT',
        'tire_age': 4
    },
    {
        'driver': 'Driver 14',
        'position': 2,
        'current_lap': 4,
        'lap_time': 95.8,  # 0.3s slower
        'tire_compound': 'SOFT',
        'tire_age': 4
    },
    {
        'driver': 'Driver 11',
        'position': 3,
        'current_lap': 4,
        'lap_time': 96.2,  # 0.7s slower
        'tire_compound': 'SOFT',
        'tire_age': 4
    }
]

# Calculate what would be different
avg_lap = (95.5 + 95.8 + 96.2) / 3

print(f"\nAverage Lap Time: {avg_lap:.2f}s")
print(f"Driver 1 delta: {95.5 - avg_lap:.3f}s (faster)")
print(f"Driver 14 delta: {95.8 - avg_lap:.3f}s (slower)")
print(f"Driver 11 delta: {96.2 - avg_lap:.3f}s (much slower)")

print("\n[NEW WEIGHTS] Position 35%, Pace 40%, Tire 15%, Model 10%:")
print("-"*70)

# Simulate what the new scoring would be
for driver_data in test_laps:
    driver = driver_data['driver']
    pos = driver_data['position']
    lap_time = driver_data['lap_time']
    
    # Position score (new)
    if pos <= 1:
        position_score = 90.0
    elif pos <= 3:
        position_score = 75.0 - (pos - 1) * 7
    elif pos <= 5:
        position_score = 60.0 - (pos - 3) * 4
    else:
        position_score = 35.0 - (pos - 10) * 1.5
    
    # Pace score (new - much better)
    pace_delta = lap_time - avg_lap
    if pace_delta < 0:
        pace_bonus = min(50.0, abs(pace_delta) / avg_lap * 600)
        pace_score = 50.0 + pace_bonus
    else:
        pace_penalty = min(40.0, pace_delta / avg_lap * 400)
        pace_score = 50.0 - pace_penalty
    
    pace_score = max(10.0, min(95.0, pace_score))
    
    # Tire (assume all same)
    tire_score = 65.0
    
    # Total with NEW weights
    total = position_score * 0.35 + pace_score * 0.40 + tire_score * 0.15 + 30 * 0.10
    
    print(f"{driver:12s} (P{pos}): Pos={position_score:.1f} | Pace={pace_score:.1f} | Total={total:.1f}%")

print("\n[KEY INSIGHT]")
print("Driver 1 now pulls ahead MORE because pace is weighted 40% (was 25%)")
print("0.3s advantage = bigger gap in final score")
