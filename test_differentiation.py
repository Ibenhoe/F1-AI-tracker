#!/usr/bin/env python3
"""Test better base confidence differentiation"""

from continuous_model_learner_v2 import ContinuousModelLearner

model = ContinuousModelLearner()

print('[TEST] Better Base Confidence Differentiation')
print('='*70)
print()

# Test LAP 11 - where we saw the problem
print('[LAP 11] Should show differentiation:')
test_data_lap11 = [
    {'driver': 'VER', 'position': 1, 'current_lap': 11, 'lap_time': 95.5, 'tire_compound': 'SOFT', 'tire_age': 4},
    {'driver': 'PER', 'position': 2, 'current_lap': 11, 'lap_time': 96.1, 'tire_compound': 'SOFT', 'tire_age': 4},
    {'driver': 'ALO', 'position': 3, 'current_lap': 11, 'lap_time': 96.8, 'tire_compound': 'SOFT', 'tire_age': 4},
    {'driver': 'NOR', 'position': 4, 'current_lap': 11, 'lap_time': 97.2, 'tire_compound': 'SOFT', 'tire_age': 4},
    {'driver': 'PIA', 'position': 5, 'current_lap': 11, 'lap_time': 97.5, 'tire_compound': 'SOFT', 'tire_age': 4},
]

for data in test_data_lap11:
    _, conf = model.predict(data)
    print(f"  {data['driver']} (P{int(data['position'])}): {conf:.1f}%")

print()
print('[LAP 4] Early race - all under 70% cap:')
test_data_lap4 = [
    {'driver': 'VER', 'position': 1, 'current_lap': 4, 'lap_time': 95.5, 'tire_compound': 'SOFT', 'tire_age': 1},
    {'driver': 'LEC', 'position': 2, 'current_lap': 4, 'lap_time': 96.1, 'tire_compound': 'SOFT', 'tire_age': 1},
    {'driver': 'NOR', 'position': 10, 'current_lap': 4, 'lap_time': 99.5, 'tire_compound': 'SOFT', 'tire_age': 1},
]

for data in test_data_lap4:
    _, conf = model.predict(data)
    print(f"  {data['driver']} (P{int(data['position'])}): {conf:.1f}%")

print()
print('[EXPECTED]')
print('  LAP 11: VER > PER > ALO > NOR > PIA (clear order)')
print('  LAP 4: VER/LEC/NOR all different but ≤ 70%')
