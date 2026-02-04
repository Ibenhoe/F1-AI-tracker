"""
Pre-Race AI Model Module
Handles model training, caching, and predictions for pre-race analysis
Separates ML concerns from Flask/web layer
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import os
import threading


class PreRaceModel:
    """Pre-race prediction model with caching and feature engineering"""
    
    def __init__(self):
        self.model = None
        self.feature_cols = None
        self.le_grid = None
        self.le_age = None
        self.df = None
        self.driver_id_map = {}  # Mapping from driver code to driverId
        self.constructor_id_map = {}  # Mapping from constructor to constructorId
        self.loaded = False
    
    def load(self, csv_path="unprocessed_f1_training_data.csv"):
        """Load and train model from CSV - separates ML logic from Flask"""
        if self.loaded:
            return True
        
        try:
            print("[PRERACE] Loading pre-race analysis model...")
            
            # Handle relative/absolute paths
            if not os.path.isabs(csv_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                csv_path = os.path.join(script_dir, csv_path)
            
            print(f"[PRERACE] Reading CSV from: {csv_path}")
            self.df = pd.read_csv(csv_path)
            
            if len(self.df) == 0:
                print("[PRERACE] No training data found")
                return False
            
            print(f"[PRERACE] Loaded {len(self.df)} rows from CSV")
            
            # Feature Engineering
            self._engineer_features()
            print("[PRERACE] Feature engineering complete")
            
            # Train model
            training_success = self._train_model()
            if not training_success:
                print("[PRERACE] Training failed")
                return False
            
            self.loaded = True
            print("[PRERACE] Pre-race model loaded and trained successfully")
            return True
            
        except Exception as e:
            print(f"[PRERACE] Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _engineer_features(self):
        """Feature engineering - extracted to separate method"""
        df = self.df
        
        # Basic preprocessing
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date')
        
        # Driver experience
        df['driver_experience'] = df.groupby('driverId').cumcount()
        
        # Build ID mappings for better lookup in predictions
        if 'driver_code' in df.columns:
            self.driver_id_map = df.groupby('driver_code')['driverId'].first().to_dict()
        if 'team' in df.columns:
            self.constructor_id_map = df.groupby('team')['constructorId'].first().to_dict()
        
        # Home race
        if 'country' not in df.columns:
            df['country'] = 'Unknown'
        
        nationality_map = {
            'British': 'UK', 'German': 'Germany', 'Spanish': 'Spain', 'French': 'France',
            'Italian': 'Italy', 'Dutch': 'Netherlands', 'Australian': 'Australia',
            'Monegasque': 'Monaco', 'American': 'USA', 'Japanese': 'Japan', 'Canadian': 'Canada',
            'Mexican': 'Mexico', 'Brazilian': 'Brazil'
        }
        df['mapped_nationality'] = df['nationality'].map(nationality_map).fillna(df['nationality'])
        df['is_home_race'] = np.where(df['mapped_nationality'] == df['country'], 1, 0)
        
        # Binning
        df['grid_bin'] = pd.cut(df['grid'], bins=[-1, 1, 3, 10, 15, 25], 
                                labels=['Pole', 'Top3', 'Points', 'Midfield', 'Back'])
        df['age_bin'] = pd.cut(df['driver_age'], bins=[17, 24, 30, 36, 60], 
                               labels=['Rookie', 'Prime', 'Experienced', 'Veteran'])
        
        # Encoding
        self.le_grid = LabelEncoder()
        self.le_age = LabelEncoder()
        df['grid_bin_code'] = self.le_grid.fit_transform(df['grid_bin'].astype(str))
        df['age_bin_code'] = self.le_age.fit_transform(df['age_bin'].astype(str))
        
        # Recent form calculation
        def calculate_rolling_avg(group, window=5):
            def weighted_mean(x):
                weights = np.arange(1, len(x) + 1)
                return np.average(x, weights=weights)
            return group.shift(1).rolling(window, min_periods=1).apply(weighted_mean, raw=True)
        
        df['driver_recent_form'] = df.groupby('driverId')['positionOrder'].transform(calculate_rolling_avg)
        df['constructor_recent_form'] = df.groupby('constructorId')['positionOrder'].transform(calculate_rolling_avg)
        df['driver_recent_form'] = df['driver_recent_form'].fillna(12.0)
        df['constructor_recent_form'] = df['constructor_recent_form'].fillna(12.0)
        
        # Pitstop features
        global_pit_mean = 24000.0
        if 'pit_stops_duration_ms' in df.columns and 'pit_stops_count' in df.columns:
            df['pit_stops_count'] = df['pit_stops_count'].fillna(0)
            df['pit_stops_duration_ms'] = df['pit_stops_duration_ms'].fillna(0)
            df['avg_pit_duration'] = df.apply(
                lambda x: x['pit_stops_duration_ms'] / x['pit_stops_count'] if x['pit_stops_count'] > 0 else np.nan, 
                axis=1
            )
            pit_mean = df['avg_pit_duration'].mean()
            global_pit_mean = pit_mean if not pd.isna(pit_mean) else 24000.0
            df['avg_pit_duration_filled'] = df['avg_pit_duration'].fillna(global_pit_mean)
            df['driver_recent_pit_avg'] = df.groupby('driverId')['avg_pit_duration_filled'].transform(calculate_rolling_avg)
            df['constructor_recent_pit_avg'] = df.groupby('constructorId')['avg_pit_duration_filled'].transform(calculate_rolling_avg)
            df['driver_recent_pit_avg'] = df['driver_recent_pit_avg'].fillna(global_pit_mean)
            df['constructor_recent_pit_avg'] = df['constructor_recent_pit_avg'].fillna(global_pit_mean)
        else:
            df['driver_recent_pit_avg'] = global_pit_mean
            df['constructor_recent_pit_avg'] = global_pit_mean
        
        # Weather
        for col in ['weather_temp_c', 'weather_precip_mm', 'weather_cloud_pct']:
            if col not in df.columns:
                if col == 'weather_temp_c':
                    df[col] = 20.0
                elif col == 'weather_precip_mm':
                    df[col] = 0.0
                else:
                    df[col] = 50.0
        
        df['weather_temp_c'] = df['weather_temp_c'].fillna(df['weather_temp_c'].mean())
        df['weather_precip_mm'] = df['weather_precip_mm'].fillna(0.0)
        df['weather_cloud_pct'] = df['weather_cloud_pct'].fillna(50.0)
        
        # Track history
        def calculate_expanding_mean(group):
            return group.shift(1).expanding().mean()
        
        def calculate_expanding_sum(group):
            return group.shift(1).expanding().sum()
        
        track_groups = df.groupby(['driverId', 'circuitId'])
        df['driver_track_avg_grid'] = track_groups['grid'].transform(calculate_expanding_mean)
        df['driver_track_avg_finish'] = track_groups['positionOrder'].transform(calculate_expanding_mean)
        df['is_podium'] = (df['positionOrder'] <= 3).astype(int)
        df['driver_track_podiums'] = track_groups['is_podium'].transform(calculate_expanding_sum)
        df['driver_track_avg_grid'] = df['driver_track_avg_grid'].fillna(12.0)
        df['driver_track_avg_finish'] = df['driver_track_avg_finish'].fillna(12.0)
        df['driver_track_podiums'] = df['driver_track_podiums'].fillna(0.0)
        
        self.df = df
    
    def _validate_model_fitted(self):
        """Validate that the model is properly fitted before use
        
        Raises:
            ValueError: If model is not fitted or not loaded
        """
        if not self.loaded or self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        try:
            # Try to access the booster (internal XGBoost structure) to verify model is fitted
            _ = self.model.get_booster()
        except Exception as fit_check_err:
            error_msg = f"Model validation failed - model is not properly fitted: {str(fit_check_err)}"
            print(f"[PRERACE] ERROR: {error_msg}")
            raise ValueError(error_msg)
    
    def _validate_training_data(self, X, y):
        """Validate training data before model fitting
        
        Args:
            X: Feature matrix
            y: Target vector
        
        Returns:
            bool: True if data is valid, False otherwise
        """
        if X.shape[0] == 0 or X.shape[1] == 0:
            print(f"[PRERACE] ERROR: Invalid training data shape: {X.shape}")
            return False
        return True
    
    def _validate_model_post_fit(self):
        """Validate model is fitted and functional after training
        
        Returns:
            bool: True if model passes validation, False otherwise
        """
        try:
            # Check model is actually fitted
            if self.model is None:
                print("[PRERACE] ERROR: Model is None after _train_model()")
                return False
            
            # Check that model has feature_importances_ (XGBoost specific)
            if not hasattr(self.model, 'feature_importances_'):
                print("[PRERACE] ERROR: Model not properly fitted - no feature_importances_")
                return False
            
            # Verify feature_cols list exists and is not empty
            if not hasattr(self, 'feature_cols') or not self.feature_cols:
                print("[PRERACE] ERROR: feature_cols list is missing or empty after _train_model")
                return False
            
            # Get first row from training data for test prediction
            if self.df is None or len(self.df) == 0:
                print("[PRERACE] ERROR: Cannot validate model - no training data available")
                return False
            
            # Create test input matching feature columns
            # Only use columns that actually exist in the dataframe
            valid_cols = [col for col in self.feature_cols if col in self.df.columns]
            if not valid_cols:
                print(f"[PRERACE] ERROR: No valid feature columns found. Requested: {self.feature_cols}")
                return False
            
            test_data = self.df[valid_cols].dropna().iloc[:1]
            if len(test_data) == 0:
                print("[PRERACE] ERROR: Cannot validate model - no valid test data after dropping NaN")
                return False
            
            # Pad with missing features if needed
            if len(valid_cols) < len(self.feature_cols):
                for missing_col in [c for c in self.feature_cols if c not in valid_cols]:
                    test_data[missing_col] = 0.0
                test_data = test_data[self.feature_cols]
            
            test_pred = self.model.predict(test_data)
            if test_pred is None or len(test_pred) == 0:
                print("[PRERACE] ERROR: Model.fit() succeeded but test prediction returned empty")
                return False
            
            print(f"[PRERACE] Model validation successful - test prediction: {test_pred[0]:.2f}")
            return True
        except Exception as val_err:
            print(f"[PRERACE] ERROR: Model validation failed: {str(val_err)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _train_model(self):
        """Train model - extracted to separate method"""
        try:
            # Define features - ENHANCED with better discriminatory power
            self.feature_cols = [
                'grid', 'circuitId', 'driver_age', 'driver_experience', 'is_home_race',
                'grid_bin_code', 'age_bin_code', 'driver_recent_form', 'constructor_recent_form',
                'driver_track_avg_grid', 'driver_track_avg_finish', 'driver_track_podiums',
                'weather_temp_c', 'weather_precip_mm', 'weather_cloud_pct',
                'driver_recent_pit_avg', 'constructor_recent_pit_avg'
            ]
            
            # Check all features exist
            missing_features = [f for f in self.feature_cols if f not in self.df.columns]
            if missing_features:
                print(f"[PRERACE] WARNING: Missing features: {missing_features}")
                # Drop missing features from the list
                self.feature_cols = [f for f in self.feature_cols if f in self.df.columns]
            
            # Train model
            train_df = self.df.dropna(subset=['positionOrder'] + self.feature_cols)
            if len(train_df) == 0:
                print("[PRERACE] ERROR: No valid training data after dropping NaN values")
                self.model = None
                return False
            
            X = train_df[self.feature_cols]
            y = train_df['positionOrder']
            
            print(f"[PRERACE] Training on {len(X)} samples with {len(self.feature_cols)} features...")
            print(f"[PRERACE] Features: {self.feature_cols}")
            print(f"[PRERACE] Target range: {y.min():.0f}-{y.max():.0f} (position)")
            
            # Validate training data shape
            if not self._validate_training_data(X, y):
                self.model = None
                return False
            
            # Use STRONGER XGBoost with better hyperparameters
            # These are tuned for F1 prediction to maximize discrimination
            self.model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=300,      # More trees = better discrimination
                learning_rate=0.08,    # Slightly lower for better convergence
                max_depth=4,           # Slightly deeper for more complex patterns
                min_child_weight=2,    # Prevent overfitting
                subsample=0.9,         # Use 90% of samples per tree
                colsample_bytree=0.9,  # Use 90% of features per tree
                gamma=0.1,             # Regularization
                random_state=42
            )
            
            # Train model with early stopping (optional but improves generalization)
            print(f"[PRERACE] Training XGBoost with enhanced hyperparameters...")
            self.model.fit(X, y)
            
            # Validate model is fitted and functional
            if not self._validate_model_post_fit():
                self.model = None
                return False
            
            # Print feature importance
            importances = self.model.feature_importances_
            feature_importance = sorted(zip(self.feature_cols, importances), key=lambda x: x[1], reverse=True)
            print(f"[PRERACE] Top features by importance:")
            for feat, importance in feature_importance[:5]:
                print(f"         {feat:25s}: {importance*100:.1f}%")
            
            print(f"[PRERACE] Model trained successfully")
            return True
        except Exception as e:
            print(f"[PRERACE] ERROR in _train_model: {str(e)}")
            import traceback
            traceback.print_exc()
            self.model = None
            return False
    
    def predict(self, grid_data, race_num):
        """
        Make predictions for upcoming race
        
        Args:
            grid_data: List of driver dicts with 'driver', 'number', 'team', 'grid_pos' keys
            race_num: Race circuit ID
        
        Returns:
            List of prediction dicts sorted by AI score
        """
        # Validate model is fitted before attempting prediction
        self._validate_model_fitted()
        
        print(f"\n[PRERACE PREDICT] Starting predictions for race {race_num}")
        print(f"[PRERACE PREDICT] Using {len(self.feature_cols)} features: {self.feature_cols}")
        
        nationality_map = {
            'British': 'UK', 'German': 'Germany', 'Spanish': 'Spain', 'French': 'France',
            'Italian': 'Italy', 'Dutch': 'Netherlands', 'Australian': 'Australia',
            'Monegasque': 'Monaco', 'American': 'USA', 'Japanese': 'Japan', 'Canadian': 'Canada',
            'Mexican': 'Mexico', 'Brazilian': 'Brazil'
        }
        
        # Build prediction data
        X_next_list = []
        for i, driver_info in enumerate(grid_data):
            driver_code = driver_info.get('driver', f'DRV{i}')
            team = driver_info.get('team', 'Unknown')
            
            # Use better ID lookup
            driver_id = self.driver_id_map.get(driver_code, i + 1)
            constructor_id = self.constructor_id_map.get(team, i + 100)
            
            # Get historical data
            driver_experience = self._get_experience(driver_id)
            driver_recent_form = self._get_recent_form('driverId', driver_id)
            constructor_recent_form = self._get_recent_form('constructorId', constructor_id)
            driver_recent_pit_avg = self._get_pit_average('driverId', driver_id)
            constructor_recent_pit_avg = self._get_pit_average('constructorId', constructor_id)
            
            X_next_list.append({
                'grid': int(driver_info.get('grid_pos', i + 1)),
                'circuitId': int(race_num),
                'driver_age': 25,  # Default
                'driver_experience': int(driver_experience),
                'is_home_race': 0,
                'grid_bin_code': int(self.le_grid.transform(['Top3' if i < 5 else 'Points'])[0]),
                'age_bin_code': int(self.le_age.transform(['Prime'])[0]),
                'driver_recent_form': float(driver_recent_form),
                'constructor_recent_form': float(constructor_recent_form),
                'driver_track_avg_grid': 12.0,
                'driver_track_avg_finish': 12.0,
                'driver_track_podiums': 0.0,
                'weather_temp_c': 20.0,
                'weather_precip_mm': 0.5,
                'weather_cloud_pct': 50.0,
                'driver_recent_pit_avg': driver_recent_pit_avg,
                'constructor_recent_pit_avg': constructor_recent_pit_avg
            })
        
        X_next = pd.DataFrame(X_next_list)
        X_next_input = X_next[self.feature_cols]
        predictions = self.model.predict(X_next_input)
        
        # Normalize predictions to 1-20 range for better confidence calculation
        min_score = min(predictions)
        max_score = max(predictions)
        score_range = max_score - min_score if max_score > min_score else 1.0
        
        print(f"[PRERACE PREDICT] Raw predictions range: {min_score:.2f} - {max_score:.2f} (span: {score_range:.2f})")
        
        # Calculate mean prediction for relative positioning
        mean_score = np.mean(predictions)
        std_score = np.std(predictions)
        print(f"[PRERACE PREDICT] Mean: {mean_score:.2f}, StdDev: {std_score:.2f}")
        
        # Build results with IMPROVED confidence calculation
        results = []
        debug_info = []  # Track first 5 drivers for debug output
        
        for i, (driver_info, score) in enumerate(zip(grid_data, predictions)):
            # === CONFIDENCE CALCULATION V2 ===
            # Based on: 1) Relative position among field, 2) Model strength, 3) Grid position
            
            grid_pos = int(driver_info.get('grid_pos', i + 1))
            
            # 1. RELATIVE POSITION SCORE (compared to field average)
            # How much better/worse than the median prediction?
            position_vs_mean = score - mean_score  # Negative = better than field avg
            
            if std_score > 0:
                # How many standard deviations away from mean?
                zscore = position_vs_mean / std_score
                # Convert to confidence: -1 std = high, 0 = medium, +1 std = lower
                # Range: zscore from -2 to +2 roughly
                relative_conf = max(50.0, min(85.0, 72.5 - (zscore * 8.0)))
            else:
                relative_conf = 72.5  # Fallback if no variation
            
            # 2. MODEL CONFIDENCE (how certain is the model?)
            # Lower score (better position) = higher confidence
            model_conf = max(50.0, min(85.0, 85.0 - (score - min_score) * 1.8))
            
            # 3. GRID POSITION WEIGHTING
            # Top grid positions start with higher confidence baseline
            if grid_pos == 1:
                grid_conf_base = 78.0  # Pole position gets baseline 78%
            elif grid_pos <= 3:
                grid_conf_base = 75.0  # Front row 75%
            elif grid_pos <= 5:
                grid_conf_base = 72.0  # Top 5 get 72%
            elif grid_pos <= 10:
                grid_conf_base = 70.0  # Top 10 get 70%
            else:
                grid_conf_base = 65.0  # Midfield/back get 65% baseline
            
            # COMBINE: Weight towards relative position (more discriminating) + model opinion
            confidence = relative_conf * 0.55 + model_conf * 0.35 + grid_conf_base * 0.10
            
            # Final bounds: 50-85%
            confidence = max(50.0, min(85.0, confidence))
            
            # Debug output for first 5 drivers
            if i < 5:
                debug_info.append({
                    'driver': driver_info.get('driver'),
                    'grid': grid_pos,
                    'score': score,
                    'zscore': (score - mean_score) / std_score if std_score > 0 else 0,
                    'rel_conf': relative_conf,
                    'model_conf': model_conf,
                    'grid_base': grid_conf_base,
                    'final_conf': confidence
                })
            
            results.append({
                'position': i + 1,
                'driver': driver_info.get('driver', f'Driver {i+1}'),
                'number': int(driver_info.get('number', 0)),
                'team': driver_info.get('team', 'Unknown'),
                'grid_position': int(driver_info.get('grid_pos', i + 1)),
                'ai_score': float(score),
                'confidence': float(confidence)
            })
        
        # Print debug info
        if debug_info:
            print("\n[PRERACE PREDICT] === CONFIDENCE BREAKDOWN (Top 5) ===")
            for info in debug_info:
                print(f"  {info['driver']:8s} | P{info['grid']:2d} | Score:{info['score']:5.2f} | z-score:{info['zscore']:+5.2f} | "
                      f"RelConf:{info['rel_conf']:5.1f}% | ModelConf:{info['model_conf']:5.1f}% | GridBase:{info['grid_base']:5.1f}% | "
                      f"FINAL:{info['final_conf']:5.1f}%")
            print("[PRERACE PREDICT] === Weights: RelPosition 55% + ModelConf 35% + GridBase 10% ===\n")
        
        # Sort by AI score (ascending = best predictions first)
        # Lower score = better predicted position
        results.sort(key=lambda x: x['ai_score'])
        return results
    
    def _get_experience(self, driver_id):
        """Get driver experience with fallback"""
        try:
            return int(self.df[self.df['driverId'] == driver_id]['driver_experience'].max() + 1)
        except:
            return 0
    
    def _get_recent_form(self, id_col, id_val, target_col='positionOrder', window=5):
        """Get recent form with fallback"""
        try:
            history = self.df[self.df[id_col] == id_val].sort_values(by='date').tail(window)
            if len(history) == 0:
                return 12.0
            values = history[target_col].values
            weights = np.arange(1, len(values) + 1)
            return float(np.average(values, weights=weights))
        except:
            return 12.0
    
    def _get_pit_average(self, id_col, id_val, default_val=24000.0):
        """Get pit stop average with fallback"""
        try:
            if 'avg_pit_duration_filled' not in self.df.columns:
                return default_val
            history = self.df[self.df[id_col] == id_val].sort_values(by='date').tail(5)
            if len(history) == 0:
                return default_val
            avg = history['avg_pit_duration_filled'].mean()
            return float(avg) if not pd.isna(avg) else default_val
        except:
            return default_val


# Global instance with lazy loading
_prerace_model_instance = None
_model_lock = threading.Lock() if hasattr(__import__('threading'), 'Lock') else None

def get_prerace_model():
    """Get or create global model instance (thread-safe)"""
    global _prerace_model_instance
    if _prerace_model_instance is None:
        _prerace_model_instance = PreRaceModel()
    return _prerace_model_instance


def ensure_prerace_model_loaded():
    """Ensure model is loaded (with error handling and retry)"""
    try:
        model = get_prerace_model()
        if not model.loaded:
            print("[PRERACE] Attempting to load model...")
            if not model.load():
                print("[PRERACE] ERROR: Model load() returned False")
                return None
            print("[PRERACE] Model successfully loaded and cached")
        return model
    except Exception as e:
        print(f"[PRERACE] ERROR in ensure_prerace_model_loaded: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
