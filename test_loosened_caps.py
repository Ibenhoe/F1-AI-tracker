#!/usr/bin/env python3
"""
Test script to verify loosened race-phase caps allow better differentiation
Expected: Max (Driver 1) should reach 75%+ range for laps 1-5 instead of being stuck at 70%
"""

from continuous_model_learner_v2 import ContinuousModelLearner
from fastf1_data_fetcher import FastF1DataFetcher
from race_predictor import predict_lap_rankings

# Load Race 5 (China)
print("[TEST] Verifying loosened caps with Race 5 (China)")
print("="*70)

fetcher = FastF1DataFetcher()
if not fetcher.fetch_race(2024, 5):
    print("Failed to load race")
    exit(1)

laps_data = fetcher.process_race_laps_streaming()

# Group by lap
laps_by_number = {}
for lap_data in laps_data:
    lap_num = int(lap_data['lap_number'])
    if lap_num not in laps_by_number:
        laps_by_number[lap_num] = []
    laps_by_number[lap_num].append(lap_data)

sorted_laps = sorted(laps_by_number.keys())

# Load model
model = ContinuousModelLearner()
model.pretrain_on_historical_data('f1_historical_5years.csv')

# Test laps 2-10
print("\n[NEW CAPS] Testing laps 2-10 with loosened race-phase caps:")
print("-"*70)
print("Expected for laps 1-5: 75% max (was 70%)")
print("Expected for laps 6-10: 82% max (was 78%)")
print("-"*70)

for lap_num in sorted_laps[1:10]:  # Laps 2-10
    if lap_num not in laps_by_number:
        continue
    
    lap_data_list = laps_by_number[lap_num]
    
    # Update model
    model.add_race_data(lap_num, lap_data_list)
    model.update_model(lap_num)
    
    # Get predictions
    top5 = predict_lap_rankings(model, lap_data_list, lap_num, 56)
    
    if top5:
        print(f"\nLAP {lap_num}:")
        for rank, pred in enumerate(top5[:3]):
            driver = pred['driver']
            accuracy = pred['accuracy']
            pos = pred['actual_pos']
            print(f"  {rank+1}. Driver {driver:3s} (Pos {pos:2.0f}): {accuracy:5.1f}% {'✓ CAN VARY' if accuracy < 73 else '✓ HAS ROOM'}")

print("\n" + "="*70)
print("[RESULT] If Max (Driver 1) shows 75%+ for lap 2-5, caps are working! ✓")
print("="*70)
