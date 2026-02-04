#!/usr/bin/env python3
"""Quick test of improved confidence calculation"""

from prerace_model import ensure_prerace_model_loaded

model = ensure_prerace_model_loaded()
if model:
    grid = [
        {'driver': 'VER', 'number': 1, 'team': 'Red Bull', 'grid_pos': 1},
        {'driver': 'LEC', 'number': 16, 'team': 'Ferrari', 'grid_pos': 2},
        {'driver': 'SAI', 'number': 55, 'team': 'Ferrari', 'grid_pos': 3},
        {'driver': 'HAM', 'number': 44, 'team': 'Mercedes', 'grid_pos': 18},
        {'driver': 'RUS', 'number': 63, 'team': 'Mercedes', 'grid_pos': 20},
    ]
    preds = model.predict(grid, 21)
    print('\n\nFINAL RESULTS:')
    for p in preds:
        print(f'{p["driver"]:8s} | Grid P{p["grid_position"]:2d} | Score: {p["ai_score"]:5.2f} | Confidence: {p["confidence"]:5.1f}%')
