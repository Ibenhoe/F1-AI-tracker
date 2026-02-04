#!/usr/bin/env python3
"""
Test script to show improved logging for race 5 (China)
"""

from prerace_model import ensure_prerace_model_loaded

# Race mapping
RACES_MAP = {
    1: "Bahrain", 2: "Saudi Arabia", 3: "Australia", 4: "Japan", 5: "China",
    6: "Miami", 7: "Monaco", 8: "Canada", 9: "Spain", 10: "Austria",
    11: "UK", 12: "Hungary", 13: "Belgium", 14: "Netherlands", 15: "Italy",
    16: "Azerbaijan", 17: "Singapore", 18: "Austin", 19: "Mexico", 20: "Brazil", 21: "Abu Dhabi"
}

def get_fallback_grid(race_num):
    """Fallback grid data with race-specific variations"""
    base_grid = [
        {'driver': 'VER', 'number': 1, 'team': 'Red Bull'},
        {'driver': 'LEC', 'number': 16, 'team': 'Ferrari'},
        {'driver': 'SAI', 'number': 55, 'team': 'Ferrari'},
        {'driver': 'PIA', 'number': 81, 'team': 'McLaren'},
        {'driver': 'NOR', 'number': 4, 'team': 'McLaren'},
        {'driver': 'HAM', 'number': 44, 'team': 'Mercedes'},
        {'driver': 'RUS', 'number': 63, 'team': 'Mercedes'},
        {'driver': 'ALO', 'number': 14, 'team': 'Aston Martin'},
        {'driver': 'STR', 'number': 18, 'team': 'Aston Martin'},
        {'driver': 'GAS', 'number': 10, 'team': 'Alpine'},
        {'driver': 'OCO', 'number': 31, 'team': 'Alpine'},
        {'driver': 'MAG', 'number': 20, 'team': 'Haas'},
        {'driver': 'HUL', 'number': 27, 'team': 'Haas'},
        {'driver': 'BOT', 'number': 77, 'team': 'Sauber'},
        {'driver': 'ZHO', 'number': 24, 'team': 'Sauber'},
        {'driver': 'TSU', 'number': 22, 'team': 'Racing Bulls'},
        {'driver': 'VER2', 'number': 23, 'team': 'Williams'},
        {'driver': 'NOR2', 'number': 25, 'team': 'Kick'},
        {'driver': 'HAM2', 'number': 50, 'team': 'Test Team'},
    ]
    
    race_adjustments = {
        1: [0, 1, -1, 0, 1, 0, -1, 1, 0, 2, -1, 1, 0, -1, 1, 0, 2, -1, 1, 0],
        2: [1, 0, 1, -1, 0, 1, 0, -1, 2, 1, 0, -1, 1, 0, -1, 2, 1, 0, -1, 1],
        3: [-1, 1, 0, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, 1, -1, 0, 1, 0, -1, 1],
        4: [0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, 1, -1, 0, 1, -1, 0, 1],
        5: [2, -1, 1, 0, -1, 1, 0, 1, -1, 0, 1, -1, 0, 1, 0, -1, 1, 0, -1, 1],
    }
    
    adjustments = race_adjustments.get(race_num, [0]*len(base_grid))
    
    grid = []
    for i, (driver, adj) in enumerate(zip(base_grid, adjustments)):
        grid_pos = i + 1 + adj
        grid_pos = max(1, min(20, grid_pos))
        grid.append({
            'driver': driver['driver'],
            'number': driver['number'],
            'team': driver['team'],
            'grid_pos': grid_pos
        })
    
    return grid

if __name__ == "__main__":
    race_num = 5
    race_name = RACES_MAP.get(race_num, "Unknown")
    
    print(f"\n{'='*80}")
    print(f"[PRERACE API] RACE {race_num}: {race_name} - Processing pre-race analysis")
    print(f"{'='*80}")
    
    print(f"\n  [GRID] Using fallback grid for race {race_num} ({race_name})")
    
    grid = get_fallback_grid(race_num)
    
    print(f"  [GRID] Top 10 drivers in fallback grid for Race {race_num} ({race_name}):")
    sorted_grid = sorted(grid, key=lambda x: x['grid_pos'])[:10]
    for driver in sorted_grid:
        print(f"    P{driver['grid_pos']:2d}: {driver['driver']:3s} - {driver['team']}")
    
    print(f"\n[PRERACE API] Total: {len(grid)} drivers loaded for Race {race_num} ({race_name})")
    
    # Load model and make predictions
    print("\nLoading model...")
    model = ensure_prerace_model_loaded()
    if model and model.loaded:
        print("✓ Model loaded")
        predictions = model.predict(grid, race_num)
        
        print(f"\n[PRERACE API] ✓ Generated {len(predictions)} predictions for Race {race_num} ({race_name})")
        print(f"[PRERACE API] Top 5 predictions:")
        for i, pred in enumerate(predictions[:5], 1):
            print(f"    {i}. {pred.get('driver'):3s} (Grid P{pred.get('grid_position'):2d}) - Confidence: {pred.get('confidence', 0):.1f}%")
        print(f"{'='*80}\n")
    else:
        print("✗ Model failed to load")
