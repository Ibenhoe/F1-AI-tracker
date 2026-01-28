"""
Race Simulator - Simulates F1 race lap-by-lap with AI predictions

ARCHITECTURAL NOTE - God Class Issue:
The current RaceSimulator class has too many responsibilities:
1. Race state management (driver states, initialization)
2. Data fetching (lap data, weather data from FastF1)
3. Lap simulation logic (real data updates vs simulated changes)
4. Driver state updates (position, tire compound, pit stops, DNF tracking)
5. Event detection (pit stops, crashes)
6. AI model integration (training model, generating predictions)
7. Output formatting (formatting driver data for WebSocket/display)

This violates Single Responsibility Principle and makes testing/maintenance difficult.

RECOMMENDED REFACTORING (Future Enhancement):
Consider splitting into focused classes:
- DriverStateManager: Manages driver positions, tires, pit stops, DNF status
- LapDataFetcher: Handles FastF1 data retrieval and weather data
- RaceEventDetector: Detects and reports race events (pit stops, crashes)
- AIModelIntegrator: Handles model training and prediction generation
- LapSimulator: Orchestrates lap-by-lap simulation logic
- RaceSimulator (simplified): Acts as orchestrator calling the above components

This would improve:
- Modularity: Each class has single clear responsibility
- Testability: Components can be tested in isolation
- Maintainability: Changes to one concern don't affect others
- Reusability: Components could be used in other applications
"""

import pandas as pd
import numpy as np
from copy import deepcopy
from datetime import timedelta


class RaceSimulator:
    """Simulates an F1 race with lap-by-lap updates and AI predictions
    
    CURRENT RESPONSIBILITIES (God Class - should be refactored):
    1. Race initialization and state management
    2. Driver state tracking (positions, tires, pit stops)
    3. Fetching real lap data from FastF1
    4. Simulating lap changes when real data unavailable
    5. AI model training on lap data
    6. Generating AI predictions for next lap
    7. Detecting race events
    8. Retrieving and formatting weather data
    9. Formatting output data for display
    
    METHODS BY RESPONSIBILITY:
    - Race init: __init__, _init_driver_states, _get_race_name, _get_total_laps
    - Data fetching: _get_lap_data, _get_weather_for_lap
    - Lap simulation: simulate_lap, _update_from_real_data, _simulate_lap_changes
    - Driver updates: _update_from_real_data (tire, position, pit stops tracking)
    - AI integration: _update_from_real_data (model training), _get_predictions
    - Event detection: _detect_events
    - Output formatting: _format_driver_output, get_current_state
    
    See module docstring for refactoring recommendations.
    """
    
    def __init__(self, race_number, model, laps_data, drivers, weather_data=None):
        """
        Initialize race simulator
        
        Args:
            race_number: F1 race number (1-21)
            model: Trained AI model (ContinuousModelLearner)
            laps_data: DataFrame with all lap data
            drivers: List of drivers in the race
            weather_data: Weather data from FastF1 session
        """
        self.race_number = race_number
        self.model = model
        self.laps_data = laps_data
        self.drivers = drivers
        self.weather_data = weather_data
        
        # Initialize driver states
        self.driver_states = {}
        self._init_driver_states()
        
        # Get race info
        self.race_name = self._get_race_name()
        self.total_laps = self._get_total_laps()
        
        print(f"[SIMULATOR] Initialized for {self.race_name} ({self.total_laps} laps)")
        
    def _init_driver_states(self):
        """Initialize driver state tracking"""
        for driver in self.drivers:
            self.driver_states[driver['code']] = {
                'driver_code': driver['code'],
                'driver_name': driver.get('name', driver['code']),
                'team': driver.get('team', 'Unknown'),
                'position': driver.get('grid_position', 20),
                'grid_position': driver.get('grid_position', 20),
                'lap_times': [],
                'current_lap_time': None,
                'tire_compound': 'SOFT',  # Default
                'tire_age': 0,  # Laps on current tire
                'pit_stops': 0,
                'dnf': False,
                'gaps': 0.0,
                'position_change': 0,
                'laps_completed': 0
            }
    
    def _get_race_name(self):
        """Get race name from data"""
        if self.laps_data is not None and len(self.laps_data) > 0:
            try:
                return f"Race {self.race_number}"
            except:
                pass
        return f"Race {self.race_number}"
    
    def _get_total_laps(self):
        """Get total race laps from data"""
        if self.laps_data is not None and len(self.laps_data) > 0:
            try:
                return int(self.laps_data['LapNumber'].max())
            except:
                pass
        return 58  # Default F1 race length
    
    def simulate_lap(self, lap_number):
        """
        Simulate a single lap by returning real FastF1 data
        
        ORCHESTRATION NOTE: This is the main orchestrator method that handles:
        1. Fetching real lap data
        2. Updating driver states
        3. Getting AI predictions
        4. Detecting events
        5. Retrieving weather
        
        Future refactoring: Delegate to focused components:
        - LapDataFetcher.get_lap_data()
        - DriverStateManager.update_lap_states()
        - AIModelIntegrator.get_predictions()
        - RaceEventDetector.detect_events()
        
        Args:
            lap_number: Current lap number
            
        Returns:
            Dictionary with lap state
        """
        lap_state = {
            'lap_number': lap_number,
            'drivers': [],
            'predictions': [],
            'events': [],
            'weather': self._get_weather_for_lap(lap_number),
            'model_metrics': None
        }
        
        # Get actual lap data for this lap from FastF1
        lap_data = self._get_lap_data(lap_number)
        
        if lap_data is not None and len(lap_data) > 0:
            # Use REAL data from FastF1
            lap_state['drivers'] = self._update_from_real_data(lap_number, lap_data)
            lap_state['predictions'] = self._get_predictions(lap_number, lap_data)
            lap_state['events'] = self._detect_events(lap_number, lap_data)
        else:
            # Only simulate if no real data available
            lap_state['drivers'] = self._simulate_lap_changes()
            lap_state['predictions'] = self._get_predictions(lap_number, None)
        
        # Add MODEL METRICS for frontend display
        if self.model is not None:
            try:
                # V2 model has lap_updates_count instead of get_model_metrics_for_frontend
                lap_state['model_metrics'] = {
                    'total_updates': self.model.lap_updates_count,
                    'model_maturity_percentage': min(100, self.model.lap_updates_count * 5),
                    'sgd_trained': self.model.sgd_model is not None,
                    'mlp_trained': self.model.mlp_model is not None,
                    'gb_trained': self.model.gb_model is not None
                }
            except Exception as e:
                print(f"[WARN] Could not get model metrics: {e}")
                lap_state['model_metrics'] = {}
        
        return lap_state
    
    def _get_lap_data(self, lap_number):
        """Get real lap data from FastF1"""
        try:
            if self.laps_data is None or len(self.laps_data) == 0:
                return None
            
            lap_filter = self.laps_data[self.laps_data['LapNumber'] == lap_number]
            return lap_filter
        except:
            return None
    
    def _update_from_real_data(self, lap_number, lap_data):
        """Update driver states from real FastF1 data
        
        RESPONSIBILITY NOTE: This method handles MULTIPLE concerns (God method):
        1. Driver state updates (position, tire age, pit stops, DNF tracking)
        2. Lap time conversion and storage
        3. AI model training on lap data
        
        Future refactoring should split this into:
        - DriverStateManager.update_from_lap_data()
        - AIModelIntegrator.train_on_lap_data()
        """
        drivers_out = []
        drivers_in_lap = set()  # Track which drivers appear in this lap
        
        for _, row in lap_data.iterrows():
            driver_code = row.get('Driver', '???')
            
            if driver_code not in self.driver_states:
                continue
            
            drivers_in_lap.add(driver_code)
            state = self.driver_states[driver_code]
            
            # Update from actual data
            if pd.notna(row.get('LapTime')):
                try:
                    lap_time = row['LapTime']
                    if hasattr(lap_time, 'total_seconds'):
                        lap_seconds = lap_time.total_seconds()
                    else:
                        lap_seconds = float(lap_time)
                    state['lap_times'].append(lap_seconds)
                    state['current_lap_time'] = f"{int(lap_seconds//60)}:{lap_seconds%60:.3f}"
                except:
                    pass
            
            # Update tire info
            if pd.notna(row.get('Compound')):
                new_compound = str(row['Compound'])
                # Only count as pit stop if tire changed AFTER first lap
                if new_compound != state['tire_compound'] and state['tire_age'] > 0:
                    state['pit_stops'] += 1
                    state['tire_age'] = 0
                    state['tire_compound'] = new_compound
                elif new_compound == state['tire_compound']:
                    state['tire_age'] += 1
                else:
                    # First lap: set the initial tire compound
                    state['tire_compound'] = new_compound
                    state['tire_age'] = 1
            
            # Update position and calculate change from GRID position
            if pd.notna(row.get('Position')):
                new_position = int(row['Position'])
                # Position change: grid_position - current_position
                # Positive = gained positions, Negative = lost positions
                state['position_change'] = state['grid_position'] - new_position
                state['position'] = new_position
            
            # Update laps completed
            state['laps_completed'] = lap_number
            
            drivers_out.append(self._format_driver_output(driver_code, state))
        
        # Check for drivers who were racing but NOT in this lap = DNF
        for driver_code, state in self.driver_states.items():
            if driver_code not in drivers_in_lap and state['laps_completed'] > 0 and not state['dnf']:
                # This driver was racing before but not in current lap = DNF!
                state['dnf'] = True
                drivers_out.append(self._format_driver_output(driver_code, state))
        
        # Sort by: active drivers first (sorted by position), then DNF drivers
        active_drivers = [d for d in drivers_out if not d['dnf']]
        dnf_drivers = [d for d in drivers_out if d['dnf']]
        
        active_drivers.sort(key=lambda x: x['position'])
        
        # Train AI model with this lap's data
        if self.model is not None:
            try:
                # Prepare lap data for training - must match pre-training features!
                # Pre-trained on: grid, grid_position, driver_age, points_constructor, circuitId, constructorId, year
                # DO NOT INCLUDE position/positionOrder in features - those are TARGETS!
                lap_training_data = []
                for driver_code, state in self.driver_states.items():
                    lap_training_data.append({
                        'grid': float(state['grid_position']),
                        'grid_position': float(state['grid_position']),
                        'driver_age': 28.0,  # Default average
                        'points_constructor': 100.0,  # Default average
                        'circuitId': 1.0,  # Placeholder
                        'constructorId': 1.0,  # Placeholder
                        'year': 2024.0,  # Current year
                        'position': float(state['position']),  # Target (separate from features)
                    })
                
                # Add to model buffer for incremental training
                self.model.lap_data_buffer.extend(lap_training_data)
                
                # Train model EVERY LAP (not every 2 laps) for continuous learning
                if len(self.model.lap_data_buffer) >= 10:
                    result = self.model.update_model(target_variable='position')
                    if result.get('status') == 'updated':
                        print(f"[MODEL] Updated - LAP {lap_number} - MAE: {result.get('mae', 0):.2f} - Updates: {self.model.lap_updates_count}/{self.model.updates_count}")
                    elif result.get('status') == 'skipped':
                        print(f"[MODEL] Skipped - LAP {lap_number}: {result.get('reason', 'Unknown')}")
                    else:
                        print(f"[MODEL] Error - LAP {lap_number}: {result.get('error', 'Unknown')}")
            except Exception as e:
                print(f"[SIMULATOR] Warning: Could not train model: {e}")
                import traceback
                traceback.print_exc()
        
        return active_drivers + dnf_drivers
    
    def _simulate_lap_changes(self):
        """Simulate lap changes when no real data available"""
        drivers_out = []
        
        for driver_code, state in self.driver_states.items():
            if state['dnf']:
                continue
            
            # Small random position fluctuations
            position_change = np.random.randint(-1, 2)
            state['position'] = max(1, min(20, state['position'] + position_change))
            state['position_change'] = position_change
            state['tire_age'] += 1  # Increment tire age each lap
            state['laps_completed'] += 1
            
            # Generate realistic lap times with tire degradation
            # Base time: 80-95 seconds depending on position and tire
            base_time = 80 + (state['position'] * 0.5)  # Faster drivers (lower position) are faster
            
            # Tire compound affects lap time
            if state['tire_compound'] == 'SOFT':
                tire_factor = 1.0  # Softs are fastest
            elif state['tire_compound'] == 'MEDIUM':
                tire_factor = 1.015  # Mediums slightly slower
            else:  # HARD
                tire_factor = 1.03  # Hards are slowest
            
            # Tire degradation: lap time gets slower as tire ages
            degradation = 1.0 + (state['tire_age'] * 0.002)  # +0.2% per lap worn
            
            # Random variation
            variation = np.random.uniform(0.95, 1.05)
            
            lap_seconds = base_time * tire_factor * degradation * variation
            state['lap_times'].append(lap_seconds)
            state['current_lap_time'] = f"{int(lap_seconds//60)}:{lap_seconds%60:.3f}"
            
            # Tire degradation: every 15 laps consider a pit stop
            if state['laps_completed'] > 1 and state['laps_completed'] % 15 == 0:
                # Pit stop roughly every 15 laps
                compounds = ['SOFT', 'MEDIUM', 'HARD']
                # Don't choose same compound as current
                available_compounds = [c for c in compounds if c != state['tire_compound']]
                state['tire_compound'] = np.random.choice(available_compounds)
                state['tire_age'] = 0  # Reset tire age after pit stop
                state['pit_stops'] += 1
            
            drivers_out.append(self._format_driver_output(driver_code, state))
        
        # Sort by position
        drivers_out.sort(key=lambda x: x['position'])
        return drivers_out
    
    def _format_driver_output(self, driver_code, state):
        """Format driver state for output"""
        return {
            'position': state['position'],
            'driver_code': driver_code,
            'driver_name': state['driver_name'],
            'team': state['team'],
            'lap_time': state['current_lap_time'] or '--:--',
            'tire_compound': state['tire_compound'],
            'tire_age': state['tire_age'],
            'pit_stops': state['pit_stops'],
            'gap': '0.000' if state['position'] == 1 else '+0.000',
            'laps_completed': state['laps_completed'],
            'position_change': state['position_change'],
            'dnf': state['dnf']
        }
    
    def _get_predictions(self, lap_number, lap_data):
        """Get AI predictions for race winner + Top 5 finishers using V2 model
        
        RESPONSIBILITY NOTE: Combines AI prediction generation with output formatting.
        Future refactoring should split into:
        - AIModelIntegrator.predict() - pure prediction logic
        - This method - formatting and aggregation only
        """
        predictions = []
        
        try:
            if self.model is None:
                # No model, return basic predictions
                for driver_code, state in sorted(
                    self.driver_states.items(),
                    key=lambda x: x[1]['position']
                )[:5]:
                    # Dynamic confidence based on position
                    base_conf = max(50, 75 - (state['position'] * 2))
                    predictions.append({
                        'position': state['position'],
                        'driver_code': driver_code,
                        'driver_name': state.get('driver_name', driver_code),
                        'confidence': base_conf,
                        'prediction': state['position'],
                        'trend': 'stable'
                    })
            else:
                # Use V2 AI model for predictions (returns tuple: pos, confidence)
                model_predictions = {}
                
                for driver_code, state in self.driver_states.items():
                    if state.get('dnf', False):
                        continue  # Skip DNF drivers
                    
                    try:
                        # Create feature dict for model.predict()
                        # MUST use EXACT SAME FEATURES as pre-training: grid, grid_position, driver_age, points_constructor, circuitId, constructorId, year
                        # Can include 'position' for context (used in confidence calculation), but NOT in scaler transform
                        
                        features = {
                            'grid': float(state['grid_position']),
                            'grid_position': float(state['grid_position']),
                            'driver_age': 28.0,  # Default average
                            'points_constructor': 100.0,  # Default average
                            'circuitId': 1.0,  # Placeholder
                            'constructorId': 1.0,  # Placeholder
                            'year': 2024.0,  # Current year
                            'position': float(state['position'])  # For context only (not used in scaler)
                        }
                        
                        # V2 model.predict() returns tuple: (predicted_position, confidence)
                        predicted_pos, confidence = self.model.predict(features)
                        
                        model_predictions[driver_code] = {
                            'position': state['position'],
                            'prediction': max(1, min(20, int(predicted_pos))),
                            'driver_code': driver_code,
                            'driver_name': state.get('driver_name', driver_code),
                            'confidence': confidence,  # V2 already caps this at max 80%
                            'trend': 'up' if predicted_pos < state['position'] else ('down' if predicted_pos > state['position'] else 'stable'),
                            'grid_pos': state['grid_position'],
                            'pit_stops': state.get('pit_stops', 0)
                        }
                    except Exception as e:
                        # Fallback: use lower confidence but still dynamic
                        fallback_conf = max(30, 60 - (state['position'] * 1.5))
                        model_predictions[driver_code] = {
                            'position': state['position'],
                            'prediction': state['position'],
                            'driver_code': driver_code,
                            'driver_name': state.get('driver_name', driver_code),
                            'confidence': fallback_conf,
                            'trend': 'stable',
                            'grid_pos': state['grid_position'],
                            'pit_stops': state.get('pit_stops', 0)
                        }
                
                # Return top 5 drivers by confidence
                predictions = sorted(
                    model_predictions.values(),
                    key=lambda x: x['confidence'],
                    reverse=True
                )[:5]
                
                # Log for debugging
                top5_strs = [f"{p['driver_code']}({p['confidence']:.0f}%)" for p in predictions]
                model_maturity = min(100, self.model.lap_updates_count * 10) if hasattr(self.model, 'lap_updates_count') else 0
                print(f"[PREDICTIONS] Lap {lap_number} - Top 5: {top5_strs} | Model maturity: {model_maturity:.0f}%")
                
        except Exception as e:
            print(f"[SIMULATOR] Error getting predictions: {str(e)}")
            # Fallback predictions based on current position
            for driver_code, state in sorted(
                self.driver_states.items(),
                key=lambda x: x[1]['position']
            )[:5]:
                if not state.get('dnf', False):
                    predictions.append({
                        'position': state['position'],
                        'driver_code': driver_code,
                        'driver_name': state.get('driver_name', driver_code),
                        'confidence': 70,
                        'prediction': state['position'],
                        'trend': 'stable',
                        'grid_pos': state['grid_position'],
                        'pit_stops': state.get('pit_stops', 0)
                    })
        
        return predictions
    
    def _detect_events(self, lap_number, lap_data):
        """Detect race events (pit stops, crashes, etc.)"""
        events = []
        
        try:
            if lap_data is None or len(lap_data) == 0:
                return events
            
            # Check for pit stops
            for _, row in lap_data.iterrows():
                if row.get('PitOutTime') is not None:
                    driver_code = row.get('Driver', '???')
                    events.append({
                        'type': 'pit_stop',
                        'driver': driver_code,
                        'lap': lap_number,
                        'message': f'{driver_code} pit stop'
                    })
        except:
            pass
        
        return events
    
    def _get_weather_for_lap(self, lap_number):
        """Get weather data for a specific lap"""
        try:
            if self.weather_data is None or len(self.weather_data) == 0:
                return None
            
            # Get weather closest to this lap
            if isinstance(self.weather_data, pd.DataFrame):
                # Get weather data for this lap (or closest available)
                weather_row = self.weather_data.iloc[-1] if len(self.weather_data) > 0 else None
                
                if weather_row is not None:
                    return {
                        'air_temp': float(weather_row.get('AirTemp', 25)) if pd.notna(weather_row.get('AirTemp')) else 25,
                        'track_temp': float(weather_row.get('TrackTemp', 35)) if pd.notna(weather_row.get('TrackTemp')) else 35,
                        'humidity': float(weather_row.get('Humidity', 50)) if pd.notna(weather_row.get('Humidity')) else 50,
                        'wind_speed': float(weather_row.get('WindSpeed', 0)) if pd.notna(weather_row.get('WindSpeed')) else 0,
                        'conditions': str(weather_row.get('Rainfall', 'Dry')) if pd.notna(weather_row.get('Rainfall')) else 'Dry'
                    }
        except Exception as e:
            print(f"[WARNING] Could not get weather data: {e}")
        
        return None
    
    def get_current_state(self):
        """Get current race state"""
        drivers = []
        for driver_code, state in self.driver_states.items():
            drivers.append(self._format_driver_output(driver_code, state))
        
        drivers.sort(key=lambda x: x['position'])
        
        return {
            'race_name': self.race_name,
            'total_laps': self.total_laps,
            'drivers': drivers,
            'weather': self._get_weather_for_lap(1)
        }
