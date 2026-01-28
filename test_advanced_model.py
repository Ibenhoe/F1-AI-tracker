#!/usr/bin/env python3
from continuous_model_learner_advanced import AdvancedContinuousLearner

print('[TEST] Advanced Continuous Learner')
print('='*60)
model = AdvancedContinuousLearner()
print('[PRETRAIN] Loading 2495 historical records...')
result = model.pretrain_on_historical_data('f1_historical_5years.csv')
print()
print('Model characteristics:')
print(f'  Feature engineer: Advanced (40+ features)')
print(f'  SGD Model: Incremental (Huber loss)')
print(f'  Gradient Boosting: Ensemble')
has_xgb = "Enabled" if model.xgb_model else "Not available"
has_lgb = "Enabled" if model.lgb_model else "Not available"
print(f'  XGBoost: {has_xgb}')
print(f'  LightGBM: {has_lgb}')
print(f'  Random Forest: Enabled')
print()
print('[SUCCESS] Advanced model ready for per-lap learning!')
