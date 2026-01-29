#!/usr/bin/env python3
"""Test RaceSimulator with AdvancedContinuousLearner integration"""

import sys
from continuous_model_learner_advanced import AdvancedContinuousLearner
from race_simulator import RaceSimulator

print('[TEST] Initializing RaceSimulator with AdvancedContinuousLearner...')
model = AdvancedContinuousLearner()
print('[OK] Model created')

# Pre-train
print('[TEST] Pre-training on historical data...')
model.pretrain_on_historical_data('f1_historical_5years.csv')
print('[OK] Pre-training complete')

# Create dummy drivers
drivers = [
    {'code': 'VER', 'name': 'Max', 'team': 'RB', 'number': 1, 'grid_position': 1},
    {'code': 'LEC', 'name': 'Charles', 'team': 'Ferrari', 'number': 16, 'grid_position': 2},
    {'code': 'SAI', 'name': 'Carlos', 'team': 'Ferrari', 'number': 55, 'grid_position': 3},
]

# Create simulator
print('[TEST] Creating RaceSimulator...')
try:
    simulator = RaceSimulator(race_number=21, model=model, laps_data=None, drivers=drivers)
    print('[OK] RaceSimulator initialized successfully')
    
    # Try to simulate one lap
    print('[TEST] Simulating lap 1...')
    state = simulator.simulate_lap(1)
    print('[OK] Lap 1 simulated successfully')
    print(f'    Drivers: {len(state["drivers"])}')
    print(f'    Predictions: {len(state["predictions"])}')
    if state['predictions']:
        top_pred = state['predictions'][0]
        print(f'    Top prediction: {top_pred["driver_code"]} ({top_pred["confidence"]:.1f}%)')
    
    # Verify structure
    if state['predictions']:
        pred = state['predictions'][0]
        required_fields = ['driver_code', 'confidence', 'trend']
        missing = [f for f in required_fields if f not in pred]
        if missing:
            print(f'    WARNING: Missing fields in prediction: {missing}')
        else:
            print(f'    Prediction structure OK')
    
    print('\n[SUCCESS] RaceSimulator + AdvancedContinuousLearner integration working!')
    
except Exception as e:
    print(f'\n[ERROR] {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
