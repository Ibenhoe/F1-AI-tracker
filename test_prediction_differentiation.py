#!/usr/bin/env python3
"""Test prediction differentiation with multiple laps and realistic grid positions"""

from continuous_model_learner_advanced import AdvancedContinuousLearner
from race_simulator import RaceSimulator

print('[TEST] Testing prediction differentiation with multiple laps...\n')

model = AdvancedContinuousLearner()
print('[SETUP] Pre-training model on 2495 historical records...')
model.pretrain_on_historical_data('f1_historical_5years.csv')
print('[OK] Model pre-trained\n')

# Create more realistic driver lineup with spread grid positions
drivers = [
    {'code': 'NOR', 'name': 'Lando Norris', 'team': 'McLaren', 'number': 4, 'grid_position': 1},
    {'code': 'RUS', 'name': 'George Russell', 'team': 'Mercedes', 'number': 63, 'grid_position': 2},
    {'code': 'LEC', 'name': 'Charles Leclerc', 'team': 'Ferrari', 'number': 16, 'grid_position': 3},
    {'code': 'SAI', 'name': 'Carlos Sainz', 'team': 'Ferrari', 'number': 55, 'grid_position': 4},
    {'code': 'HAM', 'name': 'Lewis Hamilton', 'team': 'Mercedes', 'number': 44, 'grid_position': 5},
    {'code': 'VER', 'name': 'Max Verstappen', 'team': 'Red Bull', 'number': 1, 'grid_position': 12},
    {'code': 'PER', 'name': 'Sergio Perez', 'team': 'Red Bull', 'number': 11, 'grid_position': 15},
    {'code': 'ALO', 'name': 'Fernando Alonso', 'team': 'Aston Martin', 'number': 14, 'grid_position': 8},
]

print('[SETUP] Creating RaceSimulator...')
simulator = RaceSimulator(race_number=21, model=model, laps_data=None, drivers=drivers)
print('[OK] Simulator ready\n')

print('[TEST] Simulating 10 laps and showing prediction evolution...\n')
print('=' * 100)

# Track predictions over laps
all_predictions = {}

for lap in range(1, 11):
    state = simulator.simulate_lap(lap)
    
    predictions = state['predictions']
    
    # Sort by confidence descending
    sorted_preds = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    
    # Show top 5
    print(f'\nLAP {lap}/10 PREDICTIONS (Top 5):')
    print('-' * 100)
    for rank, pred in enumerate(sorted_preds[:5], 1):
        code = pred['driver_code']
        conf = pred['confidence']
        pos = pred['position']
        bar_len = int(conf / 5)
        filled = '#' * bar_len
        empty = '-' * (20 - bar_len)
        print(f'  {rank}. {code} | Confidence: {conf:5.1f}% | Current Pos: {pos:2.0f} | [{filled}{empty}]')
    
    # Store for comparison
    for pred in sorted_preds:
        code = pred['driver_code']
        if code not in all_predictions:
            all_predictions[code] = []
        all_predictions[code].append(pred['confidence'])

print('\n' + '=' * 100)
print('\n[ANALYSIS] CONFIDENCE EVOLUTION OVER 10 LAPS:\n')

for code in sorted(all_predictions.keys()):
    confs = all_predictions[code]
    avg = sum(confs) / len(confs)
    if confs[-1] > confs[0]:
        trend = 'UP'
    elif confs[-1] < confs[0]:
        trend = 'DOWN'
    else:
        trend = 'FLAT'
    print(f'  {code}: {" ".join([f"{c:.0f}%" for c in confs])} | Avg: {avg:.1f}% [{trend}]')

print('\n[RESULT]')
if len(set([c[-1] for c in all_predictions.values()])) > 1:
    print('OK PREDICTIONS ARE DIFFERENTIATED! Different drivers have different confidence scores.')
else:
    print('Warning - Predictions still uniform - data might be too limited.')
