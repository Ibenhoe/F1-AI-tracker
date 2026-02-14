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
import warnings
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
        self.circuit_stats = None # Store circuit stats for prediction
        self.nationality_map = {}
    
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
    
    # --- HELPER FUNCTIONS FROM MAIN.PY ---
    @staticmethod
    def parse_time_str(t_str):
        """Converts '1:30.123' or '90.123' to seconds (float)."""
        if pd.isna(t_str) or str(t_str).strip() == '' or '\\N' in str(t_str): return np.nan
        try:
            t_str = str(t_str).strip()
            if ':' in t_str:
                parts = t_str.split(':')
                return float(parts[0]) * 60 + float(parts[1])
            return float(t_str)
        except:
            return np.nan

    @staticmethod
    def get_team_continuity_id(row):
        name = str(row.get('name_team', '')).lower()
        cid = row['constructorId']
        if 'mercedes' in name or 'brawn' in name or 'honda' in name or 'bar' in name: return 131
        if 'red bull' in name or 'jaguar' in name or 'stewart' in name: return 9
        if 'alpine' in name or 'renault' in name or 'lotus' in name: return 214
        if 'aston martin' in name or 'racing point' in name or 'force india' in name or 'spyker' in name or 'jordan' in name: return 117
        if 'rb' in name or 'alphatauri' in name or 'toro rosso' in name or 'minardi' in name: return 213
        if 'sauber' in name or 'alfa romeo' in name: return 15
        return cid

    @staticmethod
    def calculate_elo_history(df):
        driver_elo = {}      # Huidige rating per coureur
        constructor_elo = {} # Huidige rating per team
        
        # Opslag voor de dataframe
        d_elo_vals = []
        c_elo_vals = []
        
        # Elo parameters
        START_ELO = 1500.0
        K_FACTOR = 20 # Hoe snel verandert de rating?
        
        # We itereren per race (chronologisch)
        for race_id, group in df.groupby('raceId', sort=False):
            # 1. Huidige ratings ophalen (DIT zijn de features voor de voorspelling)
            current_d_elos = [driver_elo.get(d, START_ELO) for d in group['driverId']]
            current_c_elos = [constructor_elo.get(c, START_ELO) for c in group['team_continuity_id']]
            
            # Opslaan in de lijsten
            d_elo_vals.extend(current_d_elos)
            c_elo_vals.extend(current_c_elos)
            
            # 2. Ratings updaten op basis van uitslag
            avg_d_elo = np.mean(current_d_elos)
            avg_c_elo = np.mean(current_c_elos)
            
            count = len(group)
            for d_id, c_id, pos in zip(group['driverId'], group['team_continuity_id'], group['positionOrder']):
                actual_score = (count - pos) / (count - 1) if count > 1 else 0.5
                
                d_curr = driver_elo.get(d_id, START_ELO)
                d_exp = 1 / (1 + 10 ** ((avg_d_elo - d_curr) / 400))
                driver_elo[d_id] = d_curr + K_FACTOR * (actual_score - d_exp)
                
                c_curr = constructor_elo.get(c_id, START_ELO)
                c_exp = 1 / (1 + 10 ** ((avg_c_elo - c_curr) / 400))
                constructor_elo[c_id] = c_curr + K_FACTOR * (actual_score - c_exp)
                
        return d_elo_vals, c_elo_vals

    def _engineer_features(self):
        """Feature engineering - extracted to separate method"""
        df = self.df
        
        # --- 1. BASIC PREP ---
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

        # --- 2. ELO RATINGS (FULL HISTORY) ---
        df['team_continuity_id'] = df.apply(self.get_team_continuity_id, axis=1)
        
        # Elo berekenen op de HELE dataset
        d_elos, c_elos = self.calculate_elo_history(df)
        if len(d_elos) == len(df):
            df['driver_elo'] = d_elos
            df['constructor_elo'] = c_elos
        else:
            df['driver_elo'] = 1500.0
            df['constructor_elo'] = 1500.0

        # --- 3. TRACK CHARACTERISTICS (FULL HISTORY) ---
        if 'fastestLapSpeed' in df.columns:
            df['fastestLapSpeed_num'] = pd.to_numeric(df['fastestLapSpeed'], errors='coerce')
            
            # Circuit Type bepalen
            circuit_stats = df.groupby('circuitId')['fastestLapSpeed_num'].mean().reset_index()
            circuit_stats.rename(columns={'fastestLapSpeed_num': 'circuit_avg_speed'}, inplace=True)
            
            low_threshold = circuit_stats['circuit_avg_speed'].quantile(0.33)
            high_threshold = circuit_stats['circuit_avg_speed'].quantile(0.66)
            
            def get_track_type(speed):
                if pd.isna(speed): return 'Medium' 
                if speed < low_threshold: return 'Twisty'
                if speed > high_threshold: return 'HighSpeed'
                return 'Medium'
            
            circuit_stats['track_type'] = circuit_stats['circuit_avg_speed'].apply(get_track_type)
            self.circuit_stats = circuit_stats # Store for prediction
            df = df.merge(circuit_stats[['circuitId', 'track_type']], on='circuitId', how='left')

            # Circuit Length & Street Circuit
            if 'fastest_lap_seconds' not in df.columns:
                df['temp_seconds'] = pd.to_numeric(df['fastestLapTime'].str.replace(r'.*:', '', regex=True), errors='coerce')
            else:
                df['temp_seconds'] = df['fastest_lap_seconds']
                
            df['calc_length'] = (df['fastestLapSpeed_num'] * df['temp_seconds']) / 3600
            circuit_len = df.groupby('circuitId')['calc_length'].mean().rename('circuit_length_km')
            df = df.merge(circuit_len, on='circuitId', how='left')
            
            street_circuits = [6, 73, 15, 77, 79, 80, 1, 7] 
            df['is_street_circuit'] = df['circuitId'].isin(street_circuits).astype(int)
            df['alt'] = pd.to_numeric(df['alt'], errors='coerce').fillna(0)
        else:
            df['track_type'] = 'Medium'
            df['circuit_length_km'] = 5.0
            df['is_street_circuit'] = 0
            df['alt'] = 0

        # --- 4. FILTER 2015+ ---
        df = df[df['year'] >= 2015].copy()
        
        # Driver Experience (re-calculated after filter as per main.py logic)
        df['driver_experience'] = df.groupby('driverId').cumcount()

        # --- 5. HOME RACE ---
        if 'country' not in df.columns:
            df['country'] = 'Unknown'
        
        self.nationality_map = {
            'British': 'UK', 'German': 'Germany', 'Spanish': 'Spain', 'French': 'France',
            'Italian': 'Italy', 'Dutch': 'Netherlands', 'Australian': 'Australia',
            'Monegasque': 'Monaco', 'American': 'USA', 'Japanese': 'Japan', 'Canadian': 'Canada',
            'Mexican': 'Mexico', 'Brazilian': 'Brazil'
        }
        df['mapped_nationality'] = df['nationality'].map(self.nationality_map).fillna(df['nationality'])
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
        
        # --- 6. OVERTAKING DIFFICULTY ---
        df['pos_change_abs'] = (df['grid'] - df['positionOrder']).abs()
        df['circuit_overtake_index'] = df.groupby('circuitId')['pos_change_abs'].transform(lambda x: x.expanding().mean().shift(1))
        df['circuit_overtake_index'] = df['circuit_overtake_index'].fillna(df['pos_change_abs'].mean())

        # --- 7. RECENT FORM ---
        def calculate_rolling_avg(group, window=5):
            def weighted_mean(x):
                weights = np.arange(1, len(x) + 1)
                return np.average(x, weights=weights)
            return group.shift(1).rolling(window, min_periods=1).apply(weighted_mean, raw=True)
        
        df['driver_recent_form'] = df.groupby('driverId')['positionOrder'].transform(calculate_rolling_avg)
        df['constructor_recent_form'] = df.groupby('constructorId')['positionOrder'].transform(calculate_rolling_avg)
        df['driver_recent_form'] = df['driver_recent_form'].fillna(12.0)
        df['constructor_recent_form'] = df['constructor_recent_form'].fillna(12.0)
        
        # --- 8. PITSTOP FEATURES ---
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
            df['constructor_recent_pit_avg'] = df.groupby('team_continuity_id')['avg_pit_duration_filled'].transform(calculate_rolling_avg)
            df['driver_recent_pit_avg'] = df['driver_recent_pit_avg'].fillna(global_pit_mean)
            df['constructor_recent_pit_avg'] = df['constructor_recent_pit_avg'].fillna(global_pit_mean)
        else:
            df['driver_recent_pit_avg'] = global_pit_mean
            df['constructor_recent_pit_avg'] = global_pit_mean
        
        # --- 9. TIRE STRATEGY ---
        df['circuit_avg_stops'] = df.groupby('circuitId')['pit_stops_count'].transform(lambda x: x.expanding().mean().shift(1))
        df['circuit_avg_stops'] = df['circuit_avg_stops'].fillna(1.5)
        
        race_stint_avg = df.groupby('raceId')['avg_stint_length'].transform('mean')
        df['tire_strategy_ratio'] = df['avg_stint_length'] / race_stint_avg
        df['tire_strategy_ratio'] = df['tire_strategy_ratio'].fillna(1.0)
        df['driver_tire_strategy'] = df.groupby('driverId')['tire_strategy_ratio'].transform(calculate_rolling_avg)
        df['driver_tire_strategy'] = df['driver_tire_strategy'].fillna(1.0)

        # --- 10. RELIABILITY (DNF) ---
        if 'status' in df.columns:
            df['is_dnf'] = ~df['status'].astype(str).str.match(r'(Finished|\+\d+\sLaps)').fillna(False)
            df['is_dnf'] = df['is_dnf'].astype(int)
            df['driver_dnf_rate'] = df.groupby('driverId')['is_dnf'].transform(calculate_rolling_avg)
            df['constructor_dnf_rate'] = df.groupby('team_continuity_id')['is_dnf'].transform(calculate_rolling_avg)
            df['driver_dnf_rate'] = df['driver_dnf_rate'].fillna(0.2)
            df['constructor_dnf_rate'] = df['constructor_dnf_rate'].fillna(0.2)
        else:
            df['driver_dnf_rate'] = 0.2
            df['constructor_dnf_rate'] = 0.2

        # --- 11. QUALIFYING PACE ---
        for col in ['q1', 'q2', 'q3']:
            if col in df.columns:
                df[f'{col}_sec'] = df[col].apply(self.parse_time_str)
        
        df['best_quali_time'] = df[['q1_sec', 'q2_sec', 'q3_sec']].min(axis=1)
        pole_times = df.groupby('raceId')['best_quali_time'].min().rename('pole_time')
        df = df.merge(pole_times, on='raceId', how='left')
        df['quali_pace_deficit'] = df['best_quali_time'] / df['pole_time']
        df['quali_pace_deficit'] = df['quali_pace_deficit'].fillna(1.07)

        # --- 12. TEAMMATE BATTLE ---
        df['teammate_form_diff'] = df.groupby(['raceId', 'team_continuity_id'])['driver_recent_form'].transform(lambda x: x - x.mean())

        # --- 13. DRIVER STYLE AFFINITY ---
        df['driver_type_form'] = df.groupby(['driverId', 'track_type'])['positionOrder'].transform(
            lambda x: x.shift(1).expanding().mean()
        )
        df['driver_type_form'] = df['driver_type_form'].fillna(12.0)
        df['driver_style_affinity'] = df['driver_recent_form'] - df['driver_type_form']

        # --- 14. WEATHER ---
        for col in ['weather_temp_c', 'weather_precip_mm', 'weather_cloud_pct']:
            if col not in df.columns:
                if col == 'weather_temp_c':
                    df[col] = 20.0
                elif col == 'weather_precip_mm':
                    df[col] = 0.0
                else:
                    df[col] = 50.0
        
        df['weather_temp_c'] = df['weather_temp_c'].fillna(df['weather_temp_c'].median())
        df['weather_precip_mm'] = df['weather_precip_mm'].fillna(0.0)
        df['weather_cloud_pct'] = df['weather_cloud_pct'].fillna(50.0)
        
        # --- 15. TRACK HISTORY ---
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

        # --- 16. AGGRESSIVENESS & CONSISTENCY ---
        if 'fastestLapTime' in df.columns:
            df['fastest_lap_seconds'] = df['fastestLapTime'].apply(self.parse_time_str)
            race_best_times = df.groupby('raceId')['fastest_lap_seconds'].min().reset_index().rename(columns={'fastest_lap_seconds': 'race_best_time'})
            df = df.merge(race_best_times, on='raceId', how='left')
            df['speed_ratio'] = df['fastest_lap_seconds'] / df['race_best_time']
            df['speed_ratio'] = df['speed_ratio'].fillna(1.10)
            df['driver_recent_speed'] = df.groupby('driverId')['speed_ratio'].transform(calculate_rolling_avg)
            df['driver_recent_speed'] = df['driver_recent_speed'].fillna(1.10)
            
            df['circuit_speed_index'] = df.groupby('circuitId')['fastest_lap_seconds'].transform(lambda x: x.expanding().mean().shift(1))
            global_lap_mean = df['fastest_lap_seconds'].mean()
            df['circuit_speed_index'] = df['circuit_speed_index'].fillna(global_lap_mean)
            
            df['fastest_lap_rank'] = df.groupby('raceId')['fastest_lap_seconds'].rank(method='min')
            df['aggression_score'] = df['positionOrder'] - df['fastest_lap_rank']
            df['driver_aggression'] = df.groupby('driverId')['aggression_score'].transform(calculate_rolling_avg)
            df['driver_aggression'] = df['driver_aggression'].fillna(0.0)
            
            if 'std_lap_time_ms' in df.columns:
                df['driver_consistency'] = df.groupby('driverId')['std_lap_time_ms'].transform(calculate_rolling_avg)
                df['driver_consistency'] = df['driver_consistency'].fillna(5000.0)
            else:
                df['driver_consistency'] = 5000.0
        else:
            df['driver_recent_speed'] = 1.10
            df['circuit_speed_index'] = 90.0
            df['driver_consistency'] = 5000.0

        # --- 17. POINTS & PENALTIES ---
        df['points_before_race'] = (df.groupby(['year', 'driverId'])['points'].cumsum() - df['points']).fillna(0)
        if 'quali_position' in df.columns:
            df['quali_pos_filled'] = df['quali_position'].fillna(df['grid'])
        else:
            df['quali_pos_filled'] = df['grid']
        df['grid_penalty'] = df['grid'] - df['quali_pos_filled']

        # --- 18. WEATHER CONFIDENCE ---
        df['is_wet_race'] = df['weather_precip_mm'] > 0.1
        wet_avg = df[df['is_wet_race']].groupby('driverId')['positionOrder'].transform(lambda x: x.expanding().mean().shift(1))
        dry_avg = df[~df['is_wet_race']].groupby('driverId')['positionOrder'].transform(lambda x: x.expanding().mean().shift(1))
        df['avg_finish_wet'] = wet_avg
        df['avg_finish_dry'] = dry_avg
        df['avg_finish_dry'] = df['avg_finish_dry'].fillna(12.0)
        df['avg_finish_wet'] = df['avg_finish_wet'].fillna(df['avg_finish_dry'])
        df['weather_confidence_diff'] = df['avg_finish_wet'] - df['avg_finish_dry']

        # --- 19. OVERTAKE RATE ---
        df['positions_gained'] = df['grid'] - df['positionOrder']
        df['driver_overtake_rate'] = df.groupby('driverId')['positions_gained'].transform(calculate_rolling_avg).fillna(0.0)
        
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
        """Validate training data before model fitting (optimized for performance)
        
        Args:
            X: Feature matrix
            y: Target vector
        
        Returns:
            bool: True if data is valid, False otherwise
        """
        # Check shape first (fastest operation)
        if X.shape[0] == 0 or X.shape[1] == 0:
            print(f"[PRERACE] ERROR: Invalid training data shape: {X.shape}")
            return False
        
        # Combined validation: check NaN, Inf, and numeric types in single pass
        try:
            # Check for NaN or Inf more efficiently using pandas methods
            if X.isnull().values.any():
                nan_cols = X.columns[X.isnull().any()].tolist()
                print(f"[PRERACE] ERROR: Training data contains NaN in: {nan_cols}")
                return False
            
            # Check if all values are numeric and finite
            # Using values.any() reduces memory overhead compared to column-by-column checks
            inf_mask = np.isinf(X.select_dtypes(include=[np.number]).values)
            if inf_mask.any():
                print(f"[PRERACE] ERROR: Training data contains Inf values")
                return False
            
            # Target validation (fast)
            if y.isnull().any() or np.isinf(y.values).any():
                print(f"[PRERACE] ERROR: Target contains NaN or Inf values")
                return False
            
            # Verify all features are numeric (single dtype check per column is O(n) not O(n*m))
            non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric:
                print(f"[PRERACE] ERROR: Non-numeric features found: {non_numeric}")
                return False
            
            print(f"[PRERACE] Data validation successful: {X.shape[0]} samples, {X.shape[1]} features, all numeric")
            return True
        except Exception as val_err:
            print(f"[PRERACE] ERROR during validation: {str(val_err)}")
            return False
    
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
                'grid', 
                'grid_penalty',
                'circuitId', 
                'circuit_speed_index',
                'circuit_overtake_index',
                'driver_age',
                'driver_experience',
                'is_home_race',
                'grid_bin_code',
                'age_bin_code',
                'driver_recent_form',
                'constructor_recent_form',
                'driver_track_avg_grid',
                'driver_track_avg_finish',
                'driver_track_podiums',
                'weather_temp_c',
                'weather_precip_mm',
                'weather_cloud_pct',
                'weather_confidence_diff',
                'driver_aggression',
                'driver_overtake_rate',
                'driver_consistency',
                'driver_recent_pit_avg',
                'constructor_recent_pit_avg',
                'driver_dnf_rate',
                'constructor_dnf_rate',
                'circuit_avg_stops',
                'driver_tire_strategy',
                'driver_recent_speed',
                'points_before_race',
                'quali_pace_deficit',
                'teammate_form_diff',
                'driver_elo',
                'constructor_elo',
                'driver_style_affinity',
                'circuit_length_km',
                'is_street_circuit',
                'alt'
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
            
            # Explicitly train the model with error checking
            try:
                print(f"[PRERACE] Fitting model with X shape {X.shape} and y shape {y.shape}...")
                # Convert to numpy arrays to ensure compatibility with XGBoost
                X_np = X.values if hasattr(X, 'values') else X
                y_np = y.values if hasattr(y, 'values') else y
                
                print(f"[PRERACE] X dtype: {X_np.dtype}, y dtype: {y_np.dtype}")
                print(f"[PRERACE] X min/max: {X_np.min()}/{X_np.max()}, y min/max: {y_np.min()}/{y_np.max()}")
                
                # Fit model with numpy arrays
                self.model.fit(X_np, y_np)
                print(f"[PRERACE] Model fit completed successfully")
                print(f"[PRERACE] Model type: {type(self.model)}")
                print(f"[PRERACE] Model object id: {id(self.model)}")
                
                # Immediately check if model has feature_importances_ after fit
                has_attr = hasattr(self.model, 'feature_importances_')
                print(f"[PRERACE] Has feature_importances_ attribute: {has_attr}")
                
                if has_attr:
                    feat_imp = self.model.feature_importances_
                    print(f"[PRERACE] feature_importances_ value: {feat_imp}")
                    print(f"[PRERACE] feature_importances_ type: {type(feat_imp)}")
                    print(f"[PRERACE] feature_importances_ length: {len(feat_imp) if feat_imp is not None else 'None'}")
                    if feat_imp is None or len(feat_imp) == 0:
                        print(f"[PRERACE] ERROR: feature_importances_ is None or empty")
                        self.model = None
                        return False
                else:
                    print(f"[PRERACE] ERROR: Model.fit() returned but feature_importances_ attribute not found")
                    # Try to access booster to see if model trained
                    try:
                        booster = self.model.get_booster()
                        print(f"[PRERACE] Booster exists: {booster is not None}")
                    except Exception as e:
                        print(f"[PRERACE] Cannot get booster: {e}")
                    self.model = None
                    return False
                    
            except Exception as fit_err:
                print(f"[PRERACE] ERROR: Model.fit() failed: {str(fit_err)}")
                import traceback
                traceback.print_exc()
                self.model = None
                return False
            
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
            
            # Determine track type for this race
            track_type = 'Medium'
            if self.circuit_stats is not None:
                track_row = self.circuit_stats[self.circuit_stats['circuitId'] == int(race_num)]
                if not track_row.empty:
                    track_type = track_row.iloc[0]['track_type']
            
            # Get latest values for all features
            # Note: We use helper methods to fetch the latest known value from history
            
            # Team Continuity ID for lookups
            team_cont_id = self.get_team_continuity_id({'name_team': team, 'constructorId': constructor_id})
            
            X_next_list.append({
                'grid': int(driver_info.get('grid_pos', i + 1)),
                'grid_penalty': 0, # Assuming no penalty for prediction unless known
                'circuitId': int(race_num),
                'circuit_speed_index': self._get_latest_value('circuitId', int(race_num), 'circuit_speed_index', 90.0),
                'circuit_overtake_index': self._get_latest_value('circuitId', int(race_num), 'circuit_overtake_index', 2.0),
                'driver_age': 25,  # Default
                'driver_experience': int(driver_experience),
                'is_home_race': 0,
                'grid_bin_code': int(self.le_grid.transform(['Top3' if i < 5 else 'Points'])[0]),
                'age_bin_code': int(self.le_age.transform(['Prime'])[0]),
                'driver_recent_form': float(self._get_recent_form('driverId', driver_id)),
                'constructor_recent_form': float(self._get_recent_form('team_continuity_id', team_cont_id)),
                'driver_track_avg_grid': self._get_track_history(driver_id, int(race_num), 'grid'),
                'driver_track_avg_finish': self._get_track_history(driver_id, int(race_num), 'positionOrder'),
                'driver_track_podiums': self._get_track_history(driver_id, int(race_num), 'is_podium', agg='sum'),
                'weather_temp_c': 20.0,
                'weather_precip_mm': 0.5,
                'weather_cloud_pct': 50.0,
                'weather_confidence_diff': self._get_latest_value('driverId', driver_id, 'weather_confidence_diff', 0.0),
                'driver_aggression': self._get_recent_form('driverId', driver_id, 'aggression_score', 0.0),
                'driver_overtake_rate': self._get_recent_form('driverId', driver_id, 'positions_gained', 0.0),
                'driver_consistency': self._get_recent_form('driverId', driver_id, 'std_lap_time_ms', 5000.0),
                'driver_recent_pit_avg': self._get_pit_average('driverId', driver_id),
                'constructor_recent_pit_avg': self._get_pit_average('team_continuity_id', team_cont_id),
                'driver_dnf_rate': self._get_recent_form('driverId', driver_id, 'is_dnf', 0.2),
                'constructor_dnf_rate': self._get_recent_form('team_continuity_id', team_cont_id, 'is_dnf', 0.2),
                'circuit_avg_stops': self._get_latest_value('circuitId', int(race_num), 'circuit_avg_stops', 1.5),
                'driver_tire_strategy': self._get_recent_form('driverId', driver_id, 'tire_strategy_ratio', 1.0),
                'driver_recent_speed': self._get_recent_form('driverId', driver_id, 'speed_ratio', 1.10),
                'points_before_race': self._get_current_points(driver_id),
                'quali_pace_deficit': 1.0 + (int(driver_info.get('grid_pos', i + 1)) - 1) * 0.003, # Estimate
                'teammate_form_diff': 0.0, # Simplified
                'driver_elo': self._get_latest_value('driverId', driver_id, 'driver_elo', 1500.0),
                'constructor_elo': self._get_latest_value('team_continuity_id', team_cont_id, 'constructor_elo', 1500.0),
                'driver_style_affinity': self._get_style_affinity(driver_id, track_type),
                'circuit_length_km': self._get_latest_value('circuitId', int(race_num), 'circuit_length_km', 5.0),
                'is_street_circuit': self._get_latest_value('circuitId', int(race_num), 'is_street_circuit', 0),
                'alt': self._get_latest_value('circuitId', int(race_num), 'alt', 0)
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

    def _get_latest_value(self, id_col, id_val, target_col, default_val):
        """Get the latest known value for a column"""
        try:
            val = self.df[self.df[id_col] == id_val].sort_values(by='date').iloc[-1][target_col]
            return float(val) if not pd.isna(val) else default_val
        except:
            return default_val

    def _get_track_history(self, driver_id, circuit_id, target_col, agg='mean'):
        """Get historical performance on a specific track"""
        try:
            history = self.df[(self.df['driverId'] == driver_id) & (self.df['circuitId'] == circuit_id)]
            if len(history) == 0: return 12.0 if agg == 'mean' else 0.0
            if agg == 'sum': return float(history[target_col].sum())
            return float(history[target_col].mean())
        except:
            return 12.0 if agg == 'mean' else 0.0

    def _get_current_points(self, driver_id):
        """Get total points for current season (approximate using latest year in data)"""
        try:
            latest_year = self.df['year'].max()
            return float(self.df[(self.df['driverId'] == driver_id) & (self.df['year'] == latest_year)]['points'].sum())
        except:
            return 0.0

    def _get_style_affinity(self, driver_id, track_type):
        try:
            type_history = self.df[(self.df['driverId'] == driver_id) & (self.df['track_type'] == track_type)]
            if len(type_history) == 0: return 0.0
            avg_pos = type_history['positionOrder'].mean()
            recent_form = self._get_recent_form('driverId', driver_id)
            return float(recent_form - avg_pos)
        except:
            return 0.0


# Global instance with lazy loading (thread-safe)
_prerace_model_instance = None
_model_lock = threading.Lock()  # Always create lock, threading module is imported


def get_prerace_model():
    """Get or create global model instance (thread-safe with double-checked locking)"""
    global _prerace_model_instance
    
    # First check without lock (optimization to avoid lock contention)
    if _prerace_model_instance is not None:
        return _prerace_model_instance
    
    # Acquire lock for actual initialization
    with _model_lock:
        # Double-check inside lock in case another thread initialized it
        if _prerace_model_instance is None:
            _prerace_model_instance = PreRaceModel()
        return _prerace_model_instance


def ensure_prerace_model_loaded():
    """Ensure model is loaded (with error handling and retry, thread-safe)
    
    Note: get_prerace_model() already implements thread-safe double-checked locking
    for instance creation. This function only needs to protect the .load() call.
    """
    try:
        # get_prerace_model() is already thread-safe for instance creation
        model = get_prerace_model()
        
        # Only lock around model loading to prevent duplicate training
        with _model_lock:
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
