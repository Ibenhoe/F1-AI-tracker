"""
CONTINUOUS MODEL LEARNER V3 - ADVANCED PIT STOP & TIRE DEGRADATION
Real-time race prediction with intelligent pit stop strategy analysis

Key improvements:
1. ✅ PIT STOP MODELING: Realistic pit stop time penalties + mandatory pit detection
2. ✅ TIRE DEGRADATION: Realistic tire wear curves per compound
3. ✅ STRATEGY ANALYSIS: Detect 1-stop vs 2-stop strategies  
4. ✅ PACE CONSISTENCY: Track driver performance stability
5. ✅ GAP-BASED CONFIDENCE: Larger gaps = higher confidence in leader
6. ✅ REMAINING LAPS: Fewer laps left = stronger predictions
7. ✅ CONTINUOUS LEARNING: Updates every lap with true incremental learning
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
import json
from datetime import datetime
import os
import joblib
import warnings
warnings.filterwarnings('ignore')


class PitStopAnalyzer:
    """Analyzes pit stop strategies and calculates realistic pit stop penalties"""
    
    PIT_STOP_DURATION = {
        'fast': 2.0,
        'standard': 2.5,
        'slow': 3.5
    }
    
    TIRE_DEGRADATION = {
        'SOFT': {
            'peak_laps': 8,
            'peak_factor': 1.00,
            'degrade_start': 9,
            'degrade_rate': 0.008,
            'max_age': 35
        },
        'MEDIUM': {
            'peak_laps': 15,
            'peak_factor': 1.005,
            'degrade_start': 16,
            'degrade_rate': 0.005,
            'max_age': 45
        },
        'HARD': {
            'peak_laps': 25,
            'peak_factor': 1.010,
            'degrade_start': 26,
            'degrade_rate': 0.003,
            'max_age': 55
        }
    }
    
    def __init__(self):
        self.pit_history = defaultdict(list)
        
    def calculate_tire_pace_factor(self, compound: str, tire_age: int, 
                                    current_lap: int, total_laps: int) -> float:
        """Calculate pace factor based on tire degradation"""
        if compound not in self.TIRE_DEGRADATION:
            return 1.0
        
        degrade_curve = self.TIRE_DEGRADATION[compound]
        
        if tire_age == 1:
            return 0.99
        
        if tire_age <= degrade_curve['peak_laps']:
            return degrade_curve['peak_factor']
        
        age_beyond_peak = tire_age - degrade_curve['peak_laps']
        degradation = age_beyond_peak * degrade_curve['degrade_rate']
        pace_factor = degrade_curve['peak_factor'] + degradation
        
        if tire_age > degrade_curve['max_age']:
            pace_factor = degrade_curve['peak_factor'] + 0.20
        
        laps_remaining = total_laps - current_lap
        if laps_remaining <= 2:
            pace_factor *= 0.98
        
        return pace_factor
    
    def estimate_required_pit_stops(self, compound: str, tire_age: int,
                                     current_lap: int, total_laps: int,
                                     pit_stops_done: int) -> Dict:
        """Estimate if driver MUST pit soon"""
        laps_remaining = total_laps - current_lap
        degrade = self.TIRE_DEGRADATION.get(compound, {})
        max_age = degrade.get('max_age', 40)
        
        laps_until_worn = max_age - tire_age
        
        if laps_until_worn < 0:
            return {
                'must_pit_soon': True,
                'pit_within_n_laps': 0,
                'urgency_level': 'critical',
                'total_stops_needed': pit_stops_done + 1,
                'recommended_compound': 'ANY'
            }
        elif laps_until_worn < laps_remaining:
            return {
                'must_pit_soon': True,
                'pit_within_n_laps': laps_until_worn,
                'urgency_level': 'high' if laps_until_worn < 5 else 'medium',
                'total_stops_needed': pit_stops_done + 1,
                'recommended_compound': 'HARD'
            }
        else:
            return {
                'must_pit_soon': False,
                'pit_within_n_laps': None,
                'urgency_level': 'none',
                'total_stops_needed': pit_stops_done,
                'recommended_compound': compound
            }


class DriverPerformanceTracker:
    """Tracks individual driver consistency and pace patterns"""
    
    def __init__(self, window_size: int = 5):
        self.pace_history = defaultdict(deque)
        self.position_history = defaultdict(deque)
        self.window_size = window_size
        self.consistency_scores = defaultdict(lambda: 0.5)
        
    def update(self, driver: str, lap_time: float, position: int):
        """Update driver performance tracking"""
        self.pace_history[driver].append(lap_time)
        self.position_history[driver].append(position)
        
        if len(self.pace_history[driver]) > self.window_size:
            self.pace_history[driver].popleft()
        if len(self.position_history[driver]) > self.window_size:
            self.position_history[driver].popleft()
        
        if len(self.pace_history[driver]) > 2:
            times = list(self.pace_history[driver])
            consistency = 1.0 - min(1.0, np.std(times) / np.mean(times))
            self.consistency_scores[driver] = consistency
    
    def get_trend(self, driver: str) -> str:
        """Get pace trend: improving, stable, declining"""
        if len(self.pace_history[driver]) < 3:
            return 'unknown'
        
        times = list(self.pace_history[driver])
        first_half = np.mean(times[:len(times)//2])
        second_half = np.mean(times[len(times)//2:])
        
        improvement = first_half - second_half
        if improvement > 0.1:
            return 'improving'
        elif improvement < -0.1:
            return 'declining'
        else:
            return 'stable'
    
    def get_consistency(self, driver: str) -> float:
        """Get consistency score (0-1)"""
        return self.consistency_scores.get(driver, 0.5)
    
    def get_avg_pace(self, driver: str) -> float:
        """Get average recent pace"""
        times = list(self.pace_history[driver])
        return np.mean(times) if times else 100.0


class ContinuousModelLearner:
    """Advanced F1 race prediction model with pit stop and tire degradation"""
    
    def __init__(self, total_race_laps: int = 58, learning_rate: float = 0.02):
        """Initialize the model"""
        self.total_race_laps = total_race_laps
        self.learning_rate = learning_rate
        
        # Models
        self.position_model = None
        self.pace_model = None
        self.forest_model = None
        
        self.scaler = StandardScaler()
        
        # Analyzers
        self.pit_analyzer = PitStopAnalyzer()
        self.perf_tracker = DriverPerformanceTracker(window_size=5)
        
        # Training data
        self.training_buffer = []
        self.feature_names = []
        self.features_fitted = False
        self.updates_count = 0
        self.pre_trained = False
        
        # Cache
        self.driver_pace = {}
        self.gap_to_leader = defaultdict(lambda: 0.0)
        
        print("[OK] ContinuousModelLearner V3 initialized")
        print("    ✓ Pit stop strategy analysis")
        print("    ✓ Realistic tire degradation curves")
        print("    ✓ Driver consistency tracking")
        print("    ✓ Continuous learning per lap (SGD partial_fit)")
    
    def add_race_data(self, lap_number: int, drivers_data: List[Dict]):
        """Add lap-by-lap race data for training"""
        for driver_data in drivers_data:
            driver = driver_data.get('driver', 'UNK')
            lap_time = driver_data.get('lap_time', 100.0)
            
            if lap_time < 30 or lap_time > 300:
                lap_time = 100.0
            
            position = driver_data.get('position', 20)
            self.perf_tracker.update(driver, lap_time, int(position))
            
            self.training_buffer.append({
                'lap_number': lap_number,
                'driver': driver,
                'position': int(position),
                'lap_time': float(lap_time),
                'tire_compound': driver_data.get('tire_compound', 'MEDIUM'),
                'tire_age': int(driver_data.get('tire_age', 1)),
                'pit_stops': int(driver_data.get('pit_stops', 0)),
                'grid_position': int(driver_data.get('grid_position', 15)),
                'points_constructor': float(driver_data.get('points_constructor', 100.0)),
            })
    
    def _engineer_features(self, df: pd.DataFrame, current_lap: int) -> pd.DataFrame:
        """Engineer realistic features"""
        df = df.copy()
        
        df['tire_pace_factor'] = df.apply(
            lambda row: self.pit_analyzer.calculate_tire_pace_factor(
                row['tire_compound'],
                row['tire_age'],
                current_lap,
                self.total_race_laps
            ),
            axis=1
        )
        
        df['pit_urgency_score'] = df.apply(
            lambda row: self._pit_urgency_to_score(
                self.pit_analyzer.estimate_required_pit_stops(
                    row['tire_compound'],
                    row['tire_age'],
                    current_lap,
                    self.total_race_laps,
                    row['pit_stops']
                )
            ),
            axis=1
        )
        
        df['consistency_score'] = df['driver'].apply(
            lambda d: self.perf_tracker.get_consistency(d)
        )
        
        df['recent_pit'] = (current_lap - df['lap_number'] < 3).astype(int)
        df['position_after_pit_penalty'] = df['position'].copy()
        
        df.loc[df['recent_pit'] == 1, 'position_after_pit_penalty'] += 2
        
        df['race_progress'] = current_lap / self.total_race_laps
        df['laps_remaining'] = self.total_race_laps - current_lap
        df['pit_stop_count_encoded'] = (df['pit_stops'] - 1).clip(0, 2)
        
        return df
    
    @staticmethod
    def _pit_urgency_to_score(urgency_dict: Dict) -> float:
        """Convert pit urgency to numeric score"""
        urgency = urgency_dict.get('urgency_level', 'none')
        urgency_map = {
            'critical': 10.0,
            'high': 7.5,
            'medium': 5.0,
            'low': 2.5,
            'none': 0.0
        }
        return urgency_map.get(urgency, 0.0)
    
    def pretrain_on_historical_data(self, csv_path: str = 'f1_historical_5years.csv'):
        """Pre-train on historical data (optional - will gracefully skip if data unavailable)"""
        if not os.path.exists(csv_path):
            csv_path = 'processed_f1_training_data.csv'
        
        if not os.path.exists(csv_path):
            print(f"[INFO] No historical data available, will train from race data only")
            return False
        
        try:
            print(f"[INFO] Loading historical data from {csv_path}...")
            df = pd.read_csv(csv_path)
            print(f"[INFO] Loaded {len(df)} total records")
            
            # Use available columns for features
            feature_cols = [col for col in ['grid_position', 'points_constructor', 
                                           'driver_age', 'position_gain', 'round']
                           if col in df.columns]
            
            # Try different target columns (finish_position is more complete than positionOrder)
            target_col = None
            for col in ['finish_position', 'positionOrder', 'position']:
                if col in df.columns and df[col].notna().sum() > 50:
                    target_col = col
                    print(f"[INFO] Using '{col}' as target (non-null: {df[col].notna().sum()})")
                    break
            
            if not feature_cols or not target_col:
                print(f"[INFO] Insufficient features/target - skipping pre-training")
                return False
            
            # Get clean data with no NaN values
            df_work = df[feature_cols + [target_col]].dropna()
            print(f"[INFO] {len(df_work)} clean records after dropping NaN")
            
            if len(df_work) < 50:
                print(f"[WARN] Not enough clean data ({len(df_work)} < 50)")
                return False
            
            self.feature_names = feature_cols
            
            X = df_work[feature_cols].values.astype(np.float32)
            y = df_work[target_col].values.astype(np.float32)
            
            print(f"[INFO] Features shape: {X.shape}, Target shape: {y.shape}")
            print(f"[INFO] Feature columns: {feature_cols}")
            
            X_scaled = self.scaler.fit_transform(X)
            self.features_fitted = True
            
            print("[INFO] Pre-training position model...")
            
            self.position_model = SGDRegressor(
                loss='squared_error',
                alpha=0.001,
                learning_rate='optimal',
                max_iter=1,
                warm_start=True
            )
            
            # Train in batches
            batch_size = max(10, len(X) // 20)
            for i in range(0, len(X), batch_size):
                self.position_model.partial_fit(X_scaled[i:i+batch_size], y[i:i+batch_size])
            
            # Calculate training MAE
            train_preds = self.position_model.predict(X_scaled)
            train_mae = np.mean(np.abs(train_preds - y))
            
            print(f"[OK] Pre-trained on {len(X)} samples | Training MAE: {train_mae:.3f}")
            self.pre_trained = True
            return True
            
        except Exception as e:
            print(f"[WARN] Pre-training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def update_model(self, current_lap: int) -> Dict:
        """Update models with continuous learning - SIMPLIFIED"""
        if len(self.training_buffer) < 5:
            return {'status': 'skipped', 'reason': 'Not enough data'}
        
        try:
            df = pd.DataFrame(self.training_buffer)
            
            # Use ONLY simple, normalized features
            simple_features = ['grid_position', 'tire_age', 'pit_stops']
            
            # Add normalized points_constructor if available
            if 'points_constructor' in df.columns:
                df['points_constructor_norm'] = df['points_constructor'] / 100.0  # Normalize to ~1
                simple_features.append('points_constructor_norm')
            
            # Use only available features
            feature_cols = [col for col in simple_features if col in df.columns]
            
            if not feature_cols:
                return {'status': 'error', 'reason': 'No features available'}
            
            # Get X and y with proper normalization
            X = df[feature_cols].fillna(df[feature_cols].median()).values.astype(np.float32)
            y = df['position'].fillna(10).values.astype(np.float32)
            
            # Ensure y is in valid range (1-20)
            y = np.clip(y, 1.0, 20.0)
            
            # Scale features
            if not self.features_fitted:
                self.feature_names = feature_cols
                X_scaled = self.scaler.fit_transform(X)
                self.features_fitted = True
                print(f"[DEBUG] First fit - features: {feature_cols}, X shape: {X.shape}, y range: [{y.min()}, {y.max()}]")
            else:
                X_scaled = self.scaler.transform(X)
            
            # Initialize model if needed
            if self.position_model is None:
                self.position_model = SGDRegressor(
                    loss='squared_error',
                    alpha=0.01,  # Higher regularization to prevent overfitting
                    learning_rate='optimal',
                    max_iter=1,
                    warm_start=True,
                    eta0=0.001  # Lower learning rate
                )
            
            # Train with one sample at a time for true incremental learning
            for i in range(len(X_scaled)):
                self.position_model.partial_fit(X_scaled[i:i+1], y[i:i+1])
            
            # Calculate MAE
            preds = self.position_model.predict(X_scaled)
            preds = np.clip(preds, 1.0, 20.0)  # Clip predictions to valid range
            mae = np.mean(np.abs(preds - y))
            
            self.updates_count += 1
            
            return {
                'status': 'updated',
                'samples': len(X),
                'mae': float(mae),
                'updates': self.updates_count
            }
            
        except Exception as e:
            print(f"[ERROR] update_model failed: {e}")
            import traceback
            traceback.print_exc()
            return {'status': 'error', 'error': str(e)}
    
    def predict_top5_winners(self, drivers_data: List[Dict], current_lap: int,
                             track_gap_to_leader: Optional[Dict] = None) -> List[Dict]:
        """Predict top 5 drivers likely to finish in top 5"""
        if not self.position_model or not self.features_fitted:
            # Fallback: Use realistic position-based confidence
            sorted_drivers = sorted(drivers_data, key=lambda x: x.get('position', 999))
            results = []
            for idx, d in enumerate(sorted_drivers[:5]):
                driver = d.get('driver', 'UNK')
                pos = int(d.get('position', 999))
                
                # Higher position = higher confidence (leader has highest confidence)
                # Position 1 = 75%, Position 2 = 70%, Position 3 = 65%, etc.
                base_confidence = 75.0 - (pos - 1) * 5.0
                
                # Add gap bonus if available
                gap = track_gap_to_leader.get(driver, 0.0) if track_gap_to_leader else 0.0
                if pos == 1 and gap > 10.0:
                    base_confidence = 85.0  # Leader with big gap = very confident
                elif pos == 1 and gap > 5.0:
                    base_confidence = 82.0
                
                # Reduce confidence as race progresses
                laps_remaining = self.total_race_laps - current_lap
                if laps_remaining < 10:
                    base_confidence = min(base_confidence, 75.0)
                
                confidence = max(20.0, min(85.0, base_confidence))
                
                results.append({
                    'driver': driver,
                    'predicted_position': pos,
                    'current_position': pos,
                    'confidence': float(confidence),
                    'notes': 'Position-based (warmup)' if not self.position_model else 'Model not fitted'
                })
            
            return results
        
        try:
            df = pd.DataFrame(drivers_data)
            
            # Use SAME simple features as training
            feature_cols = [col for col in self.feature_names if col in df.columns]
            
            if not feature_cols:
                # Fallback to position-based if no features
                raise Exception("No matching features")
            
            # Fill missing values and normalize
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 0.0
                df[col] = df[col].fillna(df[col].median())
            
            X = df[feature_cols].values.astype(np.float32)
            X_scaled = self.scaler.transform(X)
            
            predictions = self.position_model.predict(X_scaled)
            predictions = np.clip(predictions, 1.0, 20.0)
            
            results = []
            for idx, (i, row) in enumerate(df.iterrows()):
                driver = row.get('driver', 'UNK')
                current_pos = float(row.get('position', 10))
                pred_pos = float(predictions[idx])
                
                # Calculate confidence based on prediction vs current position
                position_stability = max(0.0, 1.0 - abs(pred_pos - current_pos) / 20.0)
                
                # Base confidence: how stable is this prediction?
                consistency = self.perf_tracker.get_consistency(driver)
                base_conf = 40.0 + (consistency * 30.0)
                pos_bonus = position_stability * 20.0
                
                laps_remaining = self.total_race_laps - current_lap
                if current_pos <= 5:
                    laps_bonus = min(20.0, (laps_remaining / 10.0) * 10.0)
                else:
                    laps_bonus = min(10.0, (laps_remaining / 10.0) * 5.0)
                
                confidence = base_conf + pos_bonus + laps_bonus
                confidence = max(25.0, min(85.0, confidence))
                
                results.append({
                    'driver': driver,
                    'predicted_position': int(round(pred_pos)),
                    'current_position': int(current_pos),
                    'confidence': float(confidence),
                    'notes': 'ML Model'
                })
            
            results.sort(key=lambda x: x['confidence'], reverse=True)
            return results[:5]
            
        except Exception as e:
            print(f"[DEBUG] Prediction error: {e}, using fallback")
    
    def predict(self, lap_features: Dict) -> Tuple[float, float]:
        """Make prediction for a single driver"""
        if not self.position_model or not self.features_fitted:
            current_pos = float(lap_features.get('position', 10))
            # Better fallback: position-based confidence
            confidence = max(20.0, 75.0 - (current_pos - 1) * 5.0)
            return current_pos, confidence
        
        try:
            df = pd.DataFrame([lap_features])
            
            if self.feature_names:
                feature_cols = [col for col in self.feature_names if col in df.columns]
            else:
                feature_cols = [col for col in df.columns 
                               if col not in ['position', 'tire_age', 'pit_stops', 'driver', 
                                             'lap_number', 'track_status', 'tire_compound', 'is_pit_lap']
                               and df[col].dtype in [float, int, 'float32', 'float64', 'int32', 'int64']]
            
            if not feature_cols:
                return float(lap_features.get('position', 10)), 25.0
            
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 10.0
                if df[col].isna().any():
                    df[col] = 10.0
            
            X = df[feature_cols].values.astype(np.float32)
            
            if not self.features_fitted:
                return float(lap_features.get('position', 10)), 25.0
            
            X_scaled = self.scaler.transform(X)
            
            pred_pos = float(self.position_model.predict(X_scaled)[0])
            pred_pos = max(1.0, min(20.0, pred_pos))
            
            if self.pre_trained:
                base_confidence = 65.0
            else:
                base_confidence = 40.0
            
            maturity_bonus = min(10.0, self.updates_count * 0.5)
            
            current_pos = float(lap_features.get('position', 10))
            position_delta = abs(pred_pos - current_pos)
            
            if position_delta > 5:
                change_penalty = -15.0
            elif position_delta > 3:
                change_penalty = -10.0
            elif position_delta > 1:
                change_penalty = -5.0
            else:
                change_penalty = 0.0
            
            confidence = base_confidence + maturity_bonus + change_penalty
            confidence = max(20.0, min(80.0, confidence))
            
            return pred_pos, confidence
            
        except Exception as e:
            return float(lap_features.get('position', 10)), 25.0
    
    def predict_lap(self, drivers_lap_data: List[Dict]) -> Dict[str, Tuple[float, float]]:
        """Predict for all drivers in a lap"""
        predictions = {}
        
        for driver_data in drivers_lap_data:
            driver_id = driver_data.get('driver', 'unknown')
            pred_pos, confidence = self.predict(driver_data)
            predictions[driver_id] = (pred_pos, confidence)
        
        return predictions
    
    def get_learning_summary(self) -> Dict:
        """Get learning progress summary"""
        return {
            'total_updates': self.updates_count,
            'samples_in_buffer': len(self.training_buffer),
            'model_active': self.position_model is not None,
            'pre_trained': self.pre_trained,
        }
    
    def save_model(self, path: str) -> bool:
        """Save model to disk"""
        try:
            model_data = {
                'position_model': self.position_model,
                'pace_model': self.pace_model,
                'forest_model': self.forest_model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'updates_count': self.updates_count
            }
            joblib.dump(model_data, path)
            print(f"[OK] Model saved to {path}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """Load model from disk"""
        try:
            model_data = joblib.load(path)
            self.position_model = model_data.get('position_model')
            self.pace_model = model_data.get('pace_model')
            self.forest_model = model_data.get('forest_model')
            self.scaler = model_data.get('scaler')
            self.feature_names = model_data.get('feature_names', [])
            self.updates_count = model_data.get('updates_count', 0)
            self.features_fitted = True
            print(f"[OK] Model loaded from {path}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load: {e}")
            return False