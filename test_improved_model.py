#!/usr/bin/env python3
"""Quick test van verbeterd AI model met race 4 (Japan)"""

import sys
import os
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from continuous_model_learner import ContinuousModelLearner
from fastf1_data_fetcher import FastF1DataFetcher
import pandas as pd

print("="*70)
print("[TEST] Verbeterd F1 AI Model - Race 4 (Japan 2024)")
print("="*70)

# Initialize model
model = ContinuousModelLearner(learning_decay=0.85, min_samples_to_train=15)

# Pre-train on NEW historical data
print("\n[PRETRAIN] Loading historical F1 data (2020-2024)...")
if os.path.exists('f1_historical_5years.csv'):
    print("  [LOAD] Using f1_historical_5years.csv (1025+ records)")
    result = model.pretrain_on_historical_data('f1_historical_5years.csv', exclude_race_number=4, current_year=2024)
    print(f"  [RESULT] Pre-training: {result}")
else:
    print("  [FALLBACK] Using processed_f1_training_data.csv")
    model.pretrain_on_historical_data('processed_f1_training_data.csv')

print(f"\n[MODEL] Pre-trained state: {model.pre_trained}")
print(f"[MODEL] Model exists: {model.model is not None}")

# Fetch race data
print("\n[FETCH] Loading race 4 (Japan) data...")
fetcher = FastF1DataFetcher()
if not fetcher.fetch_race(2024, 4):
    print("[ERROR] Failed to fetch race")
    sys.exit(1)

laps = fetcher.get_laps()
if not laps or len(laps) == 0:
    print("[ERROR] No lap data fetched")
    sys.exit(1)

print(f"[OK] Got {len(laps)} laps")

# Get unique lap numbers
laps_by_number = {}
for _, lap in laps.iterrows():
    lap_num = int(lap['LapNumber'])
    if lap_num not in laps_by_number:
        laps_by_number[lap_num] = []
    laps_by_number[lap_num].append(lap)

# Show predictions for lap 5 and lap 30
for target_lap in [5, 30, 53]:
    if target_lap not in laps_by_number:
        print(f"\n[LAP-{target_lap}] Not available")
        continue
    
    print(f"\n[LAP-{target_lap}] ========================================")
    lap_data_list = laps_by_number[target_lap]
    
    # Build drivers_lap_data format
    drivers_lap_data = []
    for lap in lap_data_list:
        drivers_lap_data.append({
            'driver': lap['Driver'],
            'position': lap['Position'],
            'lap_time': lap['Time'].total_seconds() if pd.notna(lap['Time']) else None,
            'pace': 100.0,  # Simplified
        })
    
    # Get predictions
    result = model.predict_winner(drivers_lap_data)
    predictions = result['predictions']
    
    # Show top 5 predictions
    sorted_preds = sorted(predictions.items(), 
                         key=lambda x: x[1].get('predicted_position', 99))
    
    for rank, (driver, pred) in enumerate(sorted_preds[:5], 1):
        curr_pos = pred['current_position']
        pred_pos = pred['predicted_position']
        conf = pred['confidence']
        print(f"  [{rank}] {driver:3s} - Current P{curr_pos:.0f} -> Predicted P{pred_pos:.0f} [Confidence: {conf:.0f}%]")

print("\n" + "="*70)
print("[OK] Test complete!")
