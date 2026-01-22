"""
Simple test to verify improved model works
"""
import sys
sys.path.insert(0, '.')

from continuous_model_learner import ContinuousModelLearner
from fastf1_data_fetcher import FastF1DataFetcher

print("="*70)
print("[TEST] Simple Model Test - Race 4 Japan 2024")
print("="*70)

try:
    # Initialize model
    print("\n[1] Loading model...")
    model = ContinuousModelLearner()
    print("  [OK] Model loaded")
    
    # Pre-train with 2024 data
    print("\n[2] Pre-training with historical data...")
    model.pretrain_on_historical_data(
        'f1_historical_5years.csv',
        exclude_race_number=4,
        current_year=2024
    )
    print("  [OK] Pre-training complete")
    
    # Fetch race data
    print("\n[3] Fetching race 4 data...")
    fetcher = FastF1DataFetcher()
    if not fetcher.fetch_race(2024, 4):
        print("  [ERROR] Failed to fetch race")
        sys.exit(1)
    
    laps = fetcher.get_laps_data()
    race_info = {
        'event_name': fetcher.session.event['EventName'],
        'location': fetcher.session.event['Location'],
        'drivers': fetcher.get_drivers_in_race()
    }
    print(f"  [OK] Loaded {len(laps)} laps")
    
    # Get final predictions
    print("\n[4] Getting predictions...")
    predictions = model.predict_winner(
        laps,
        race_info=race_info,
        verbose=False
    )
    print(f"  [OK] Got {len(predictions)} driver predictions")
    
    # Show top 5
    print("\n[5] TOP 5 PREDICTIONS:")
    for i, pred in enumerate(predictions[:5], 1):
        print(f"  {i}. P{pred['position']:2d} - {pred['driver_code']:3s} "
              f"({pred['confidence']*100:.0f}%) "
              f"Position change: {pred.get('position_delta', 0):+d}")
    
    print("\n" + "="*70)
    print("[OK] Test successful!")
    print("="*70)

except Exception as e:
    print(f"\n[ERROR] {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()
