"""
CONTINUOUS MODEL LEARNER V2 - TRUE INCREMENTAL LEARNING PER LAP
Real-time model training with partial_fit every lap

Dit systeem:
1. Verzamelt lap data per lap (alle 20 drivers)
2. Traints model incrementeel per lap met partial_fit (SGDRegressor + MLPRegressor)
3. Model verbetert zich REAL-TIME naarmate race vordert
4. Confidence scores worden dynamisch berekend (NOOIT 100%)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import os


class ContinuousModelLearner:
    """
    Real-time model updater met ECHTE continuous learning per lap
    Gebruikt SGDRegressor voor true incremental partial_fit per lap
    """
    
    def __init__(self, learning_decay: float = 0.95, min_samples_to_train: int = 3):
        """
        Args:
            learning_decay: Decay factor voor sample weights (nieuwe laps krijgen meer gewicht)
            min_samples_to_train: Minimaal samples voor model update (LAAG voor per-lap)
        """
        self.learning_decay = learning_decay
        self.min_samples_to_train = min_samples_to_train
        
        # ===== INCREMENTAL LEARNING MODELS =====
        self.sgd_model = None  # SGDRegressor - true partial_fit support
        self.mlp_model = None  # MLPRegressor - ensemble approach
        self.gb_model = None   # GradientBoostingRegressor - fallback
        
        self.scaler = StandardScaler()
        self.feature_names = []
        self.features_fitted = False
        
        # Training tracking
        self.training_history = []
        self.lap_data_buffer = []
        self.updates_count = 0
        self.lap_updates_count = 0  # Track lap-by-lap updates
        self.pre_trained = False
        self.last_predictions = {}  # Cache predictions
        
        print("[OK] ContinuousModelLearner V2 initialized (Incremental Learning Mode)")
        print("    - Using SGDRegressor for true partial_fit per lap")
        print("    - Using MLPRegressor for ensemble predictions")
        print("    - Model updates EVERY lap with new data")
        print("    - Confidence capped at max 80% (NEVER 100%)")
    
    def pretrain_on_historical_data(self, csv_path: str = 'f1_historical_5years.csv',
                                     exclude_race_number: Optional[int] = None,
                                     current_year: Optional[int] = None):
        """
        Pre-train model incrementally on historical F1 data using partial_fit
        Dit geeft het model een baseline maar het zal nog steeds per lap verbeteren
        
        Args:
            csv_path: Path naar CSV met historische race data
        """
        if not os.path.exists(csv_path):
            print(f"[WARN] {csv_path} not found, trying fallback...")
            csv_path = 'processed_f1_training_data.csv'
        
        if not os.path.exists(csv_path):
            print(f"[WARN] No historical data - model will learn from scratch per lap")
            return False
        
        try:
            print(f"[INFO] Loading historical data from {csv_path} for pre-training...")
            df = pd.read_csv(csv_path)
            print(f"[OK] Loaded {len(df)} historical race records")
            
            # Identify available columns
            available_cols = df.columns.tolist()
            
            # Select features
            possible_features = [
                'grid', 'grid_position',
                'driver_age', 
                'points_constructor',
                'circuitId', 
                'constructorId',
                'year'
            ]
            
            feature_cols = [col for col in possible_features if col in available_cols]
            
            # Find target
            target_col = None
            for candidate in ['positionOrder', 'finish_position', 'position']:
                if candidate in available_cols:
                    target_col = candidate
                    break
            
            if not target_col or not feature_cols:
                print(f"[WARN] Insufficient data in CSV, skipping pre-training")
                return False
            
            print(f"[OK] Using features: {feature_cols}")
            print(f"[OK] Target: {target_col}")
            
            # Prepare data
            df_work = df[feature_cols + [target_col]].copy()
            
            # Fill NaN with median
            for col in feature_cols:
                if df_work[col].isna().any():
                    df_work[col].fillna(df_work[col].median(), inplace=True)
            if df_work[target_col].isna().any():
                df_work[target_col].fillna(df_work[target_col].median(), inplace=True)
            
            df_clean = df_work.dropna()
            if len(df_clean) < 30:
                print(f"[WARN] Too few records ({len(df_clean)}) for pre-training")
                return False
            
            X = df_clean[feature_cols].values.astype(np.float32)
            y = df_clean[target_col].values.astype(np.float32)
            
            # Store feature names for later use
            self.feature_names = feature_cols
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            self.features_fitted = True
            
            # ===== INCREMENTAL PRE-TRAINING =====
            print("[INFO] Pre-training models incrementally on historical data...")
            
            # Initialize SGDRegressor (true online learning)
            self.sgd_model = SGDRegressor(
                loss='squared_error',
                learning_rate='optimal',
                eta0=0.01,
                alpha=0.001,  # L2 regularization
                random_state=42,
                warm_start=True,
                max_iter=1
            )
            
            # Train in batches (simulates incremental learning)
            batch_size = min(50, max(10, len(X) // 20))
            num_batches = (len(X) + batch_size - 1) // batch_size
            print(f"    Training SGDRegressor in {num_batches} batches...")
            for i in range(0, len(X), batch_size):
                X_batch = X_scaled[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                self.sgd_model.partial_fit(X_batch, y_batch)
            
            print(f"[OK] SGDRegressor pre-trained on {len(X)} samples")
            
            # Initialize MLPRegressor as ensemble
            print(f"    Training MLPRegressor...")
            self.mlp_model = MLPRegressor(
                hidden_layer_sizes=(32, 16),
                learning_rate='adaptive',
                max_iter=100,
                alpha=0.001,
                random_state=42,
                warm_start=True,
                early_stopping=False
            )
            self.mlp_model.fit(X_scaled, y)
            print(f"[OK] MLPRegressor pre-trained on {len(X)} samples")
            
            # Initialize GradientBoosting as fallback
            print(f"    Training GradientBoostingRegressor...")
            self.gb_model = GradientBoostingRegressor(
                n_estimators=50,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                random_state=42
            )
            self.gb_model.fit(X_scaled, y)
            print(f"[OK] GradientBoostingRegressor pre-trained on {len(X)} samples")
            
            self.pre_trained = True
            self.updates_count = 100
            print(f"\n[OK] Pre-training complete!")
            print(f"    Model will continue learning incrementally per lap during race")
            
            return True
            
        except Exception as e:
            print(f"[WARN] Pre-training failed: {e}")
            print(f"       Model will learn from scratch during race (lap-by-lap)")
            return False

    
    def add_lap_data(self, lap_number: int, drivers_data: List[Dict]):
        """
        Voeg lap data toe van alle drivers
        
        Args:
            lap_number: Lap nummer in de race
            drivers_data: List van driver lap features dictionaries
        """
        for driver_data in drivers_data:
            self.lap_data_buffer.append({
                'lap_number': lap_number,
                **driver_data
            })
    
    
    def prepare_training_data(self, target_variable: str = 'position') -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Bereid training data voor voor incremental learning
        
        Args:
            target_variable: Target to predict ('position')
            
        Returns:
            (X_scaled, y) or (None, None) if not enough data
        """
        if len(self.lap_data_buffer) < self.min_samples_to_train:
            return None, None
        
        df = pd.DataFrame(self.lap_data_buffer)
        
        # Use EXACT same features as pre-training (critical for scaler compatibility!)
        if self.feature_names:
            # Use pre-trained feature order
            feature_cols = [col for col in self.feature_names if col in df.columns]
        else:
            # Select numeric features only (fallback)
            feature_cols = [col for col in df.columns 
                           if col not in ['driver', 'lap_number', target_variable, 'track_status', 
                                         'tire_compound', 'is_pit_lap', 'position', 'positionOrder']
                           and df[col].dtype in [float, int, 'float32', 'float64', 'int32', 'int64']]
            
            if not feature_cols:
                # Fall back to using known features
                feature_cols = [col for col in ['grid', 'driver_age', 'points_constructor', 
                                                'circuitId', 'constructorId']
                               if col in df.columns]
        
        if not feature_cols or target_variable not in df.columns:
            return None, None
        
        # Fill NaN
        for col in feature_cols:
            df[col].fillna(df[col].median(), inplace=True)
        df[target_variable].fillna(df[target_variable].median(), inplace=True)
        
        X = df[feature_cols].values.astype(np.float32)
        y = df[target_variable].values.astype(np.float32)
        
        # Scale
        if not self.features_fitted:
            self.feature_names = feature_cols
            X_scaled = self.scaler.fit_transform(X)
            self.features_fitted = True
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y
    
    
    def update_model(self, target_variable: str = 'position') -> Dict:
        """
        Update model incrementally with new lap data using partial_fit
        THIS IS THE KEY CONTINUOUS LEARNING METHOD!
        
        Args:
            target_variable: Target to predict
            
        Returns:
            Dict with update metrics
        """
        # Prepare data
        X_scaled, y = self.prepare_training_data(target_variable)
        
        if X_scaled is None:
            return {
                'status': 'skipped',
                'reason': f'Not enough data: {len(self.lap_data_buffer)} samples (need {self.min_samples_to_train})'
            }
        
        try:
            # ===== PARTIAL FIT - TRUE INCREMENTAL LEARNING =====
            
            # Update SGDRegressor with partial_fit (online learning)
            if self.sgd_model is None:
                self.sgd_model = SGDRegressor(
                    loss='squared_error',
                    learning_rate='optimal',
                    eta0=0.01,
                    alpha=0.001,
                    random_state=42,
                    warm_start=True,
                    max_iter=1
                )
            
            self.sgd_model.partial_fit(X_scaled, y)
            
            # Update MLPRegressor with partial_fit
            if self.mlp_model is None:
                self.mlp_model = MLPRegressor(
                    hidden_layer_sizes=(32, 16),
                    learning_rate='adaptive',
                    max_iter=50,
                    alpha=0.001,
                    random_state=42,
                    warm_start=True,
                    early_stopping=False
                )
                self.mlp_model.fit(X_scaled, y)
            else:
                # Re-train on accumulated data (warm_start preserves learning)
                self.mlp_model.max_iter = 5  # Few iterations per update
                self.mlp_model.fit(X_scaled, y)
            
            # Update GradientBoosting (rebuild with more data)
            if self.gb_model is None:
                self.gb_model = GradientBoostingRegressor(
                    n_estimators=30,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    random_state=42
                )
                self.gb_model.fit(X_scaled, y)
            else:
                # Rebuild with more estimators as we get more data
                n_est = min(100, 30 + (self.lap_updates_count // 5))
                self.gb_model = GradientBoostingRegressor(
                    n_estimators=n_est,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    random_state=42
                )
                self.gb_model.fit(X_scaled, y)
            
            self.lap_updates_count += 1
            self.updates_count += 1
            
            # Calculate metrics
            sgd_preds = self.sgd_model.predict(X_scaled)
            mse = np.mean((sgd_preds - y) ** 2)
            mae = np.mean(np.abs(sgd_preds - y))
            
            result = {
                'status': 'updated',
                'lap_update': self.lap_updates_count,
                'total_updates': self.updates_count,
                'samples_used': len(X_scaled),
                'mse': float(mse),
                'mae': float(mae),
                'model_count': sum([self.sgd_model is not None, 
                                   self.mlp_model is not None,
                                   self.gb_model is not None])
            }
            
            self.training_history.append(result)
            return result
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    
    def predict(self, lap_features: Dict) -> Tuple[float, float]:
        """
        Maak voorspelling van position met ENSEMBLE van alle models
        Returns: (predicted_position, confidence)
        
        NOOIT 100% confidence - capped at 80%!
        lap_features mag EXTRA velden bevatten (position, tire_age, pit_stops) 
        maar die worden niet als features gebruikt - alleen voor context!
        """
        if not (self.sgd_model or self.mlp_model or self.gb_model):
            # No model trained yet - return current position with low confidence
            current_pos = float(lap_features.get('position', 10))
            return current_pos, 25.0  # 25% confidence when no model
        
        try:
            # Get feature vector - EXCLUDE non-feature fields like position, tire_age, pit_stops
            df = pd.DataFrame([lap_features])
            
            # Use pre-trained feature names (which exclude position/tire_age/pit_stops)
            if self.feature_names:
                feature_cols = [col for col in self.feature_names if col in df.columns]
            else:
                feature_cols = [col for col in df.columns 
                               if col not in ['position', 'tire_age', 'pit_stops', 'driver', 
                                             'lap_number', 'track_status', 'tire_compound', 'is_pit_lap']
                               and df[col].dtype in [float, int, 'float32', 'float64', 'int32', 'int64']]
            
            if not feature_cols:
                return float(lap_features.get('position', 10)), 25.0
            
            # Fill missing features
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 10.0
                if df[col].isna().any():
                    df[col] = 10.0
            
            X = df[feature_cols].values.astype(np.float32)
            
            if not self.features_fitted:
                return float(lap_features.get('position', 10)), 25.0
            
            X_scaled = self.scaler.transform(X)
            
            # ENSEMBLE PREDICTIONS
            predictions = []
            
            if self.sgd_model is not None:
                pred_sgd = self.sgd_model.predict(X_scaled)[0]
                predictions.append(pred_sgd)
            
            if self.mlp_model is not None:
                pred_mlp = self.mlp_model.predict(X_scaled)[0]
                predictions.append(pred_mlp)
            
            if self.gb_model is not None:
                pred_gb = self.gb_model.predict(X_scaled)[0]
                predictions.append(pred_gb)
            
            # Average ensemble prediction
            if predictions:
                pred_pos = float(np.mean(predictions))
            else:
                pred_pos = float(lap_features.get('position', 10))
            
            # Clamp to valid range
            pred_pos = max(1.0, min(20.0, pred_pos))
            
            # ===== REALISTIC CONFIDENCE CALCULATION =====
            # Base confidence: lower when just started, higher when pre-trained
            if self.pre_trained:
                base_confidence = 65.0  # 65% baseline when pre-trained
            else:
                base_confidence = 40.0  # 40% when learning from scratch
            
            # Add bonus for model maturity (more updates = slightly more confident)
            maturity_bonus = min(10.0, self.lap_updates_count * 0.5)
            
            # Penalty for big position changes (uncertain if changing lots)
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
            
            # Add variance penalty (more models agreeing = higher confidence)
            if len(predictions) >= 2:
                pred_variance = np.var(predictions)
                if pred_variance < 0.5:
                    variance_bonus = 5.0
                elif pred_variance < 1.0:
                    variance_bonus = 2.0
                else:
                    variance_bonus = -3.0
            else:
                variance_bonus = 0.0
            
            # FINAL CONFIDENCE
            confidence = base_confidence + maturity_bonus + change_penalty + variance_bonus
            
            # ===== HARD CAP: NEVER 100% =====
            # Maximum 80% confidence even with perfect models
            confidence = max(20.0, min(80.0, confidence))
            
            return pred_pos, confidence
            
        except Exception as e:
            # Silent fallback
            return float(lap_features.get('position', 10)), 25.0
    
    
    def predict_lap(self, drivers_lap_data: List[Dict]) -> Dict[str, Tuple[float, float]]:
        """
        Voorspel position voor alle drivers in een lap
        
        Returns:
            Dict: {driver_id: (predicted_pos, confidence), ...}
        """
        predictions = {}
        
        for driver_data in drivers_lap_data:
            driver_id = driver_data.get('driver', 'unknown')
            pred_pos, confidence = self.predict(driver_data)
            predictions[driver_id] = (pred_pos, confidence)
        
        return predictions
    
    
    def get_learning_summary(self) -> Dict:
        """
        Get summary van continuous learning progress
        """
        return {
            'lap_updates': self.lap_updates_count,
            'total_updates': self.updates_count,
            'samples_in_buffer': len(self.lap_data_buffer),
            'models_active': sum([self.sgd_model is not None, 
                                 self.mlp_model is not None,
                                 self.gb_model is not None]),
            'pre_trained': self.pre_trained,
            'recent_history': self.training_history[-5:] if self.training_history else []
        }
    
    
    def save_model(self, path: str) -> bool:
        """
        Sla models op naar bestand
        """
        try:
            model_data = {
                'sgd_model': self.sgd_model,
                'mlp_model': self.mlp_model,
                'gb_model': self.gb_model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'updates_count': self.updates_count
            }
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"[OK] Models saved to: {path}")
            return True
        except Exception as e:
            print(f"[ERROR] Error saving models: {e}")
            return False
    
    
    def load_model(self, path: str) -> bool:
        """
        Laad models van bestand
        """
        try:
            import pickle
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            self.sgd_model = model_data.get('sgd_model')
            self.mlp_model = model_data.get('mlp_model')
            self.gb_model = model_data.get('gb_model')
            self.scaler = model_data.get('scaler')
            self.feature_names = model_data.get('feature_names', [])
            self.updates_count = model_data.get('updates_count', 0)
            self.features_fitted = True
            print(f"[OK] Models loaded from: {path}")
            return True
        except Exception as e:
            print(f"[ERROR] Error loading models: {e}")
            return False
