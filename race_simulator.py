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
from battle_detector import BattleDetector
from event_generator import RaceEventGenerator


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
        
        # Track previous driver state for event detection (pit stops, tire changes)
        self.previous_driver_state = {}
        
        # Get race info
        self.race_name = self._get_race_name()
        self.total_laps = self._get_total_laps()
        
        # Initialize event generation system
        self.battle_detector = BattleDetector()
        self.event_generator = RaceEventGenerator()
        
        # DEBUG: Show which laps are available in the data
        if self.laps_data is not None and len(self.laps_data) > 0:
            try:
                unique_laps = sorted(self.laps_data['LapNumber'].unique())
                print(f"[SIMULATOR] Available laps in data: {unique_laps[:10]}{'...' if len(unique_laps) > 10 else ''}")
                print(f"[SIMULATOR] Total rows in laps_data: {len(self.laps_data)}")
                
                # Show which drivers are in lap 1 and lap 2
                if len(unique_laps) > 0:
                    lap1_data = self.laps_data[self.laps_data['LapNumber'] == unique_laps[0]]
                    print(f"[SIMULATOR] Lap {unique_laps[0]}: {len(lap1_data)} driver records")
                    
                    if len(unique_laps) > 1:
                        lap2_data = self.laps_data[self.laps_data['LapNumber'] == unique_laps[1]]
                        print(f"[SIMULATOR] Lap {unique_laps[1]}: {len(lap2_data)} driver records")
            except Exception as e:
                print(f"[SIMULATOR] Could not analyze lap data: {e}")
        else:
            print(f"[SIMULATOR] WARNING: No lap data provided!")
        
        print(f"[SIMULATOR] Initialized for {self.race_name} ({self.total_laps} laps)")
        print(f"[SIMULATOR] Event system enabled - battles & pit stops will be tracked")
        
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
                'gap_to_leader': 0.0,  # Gap to leader for battle detection
                'gap_to_next': 0.0,    # Gap to next driver
                'cumulative_gap': 0.0, # Accumulated gap to LEADER (grows each lap!)
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
        4. Detecting events (battles, pit stops, etc.)
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
            print(f"[LAP {lap_number}] ✓ Real FastF1 data available ({len(lap_data)} drivers)")
            
            # IMPORTANT: Update drivers first (which trains the model), THEN get predictions
            # This ensures predictions use the updated model with current lap's data
            lap_state['drivers'] = self._update_from_real_data(lap_number, lap_data)
            lap_state['predictions'] = self._get_predictions(lap_number, lap_data)
            
            # DEBUG: Show first 3 drivers for debugging (every 3 laps)
            if lap_number > 0 and lap_number % 3 == 0 and len(lap_state['drivers']) > 0:
                top_3 = lap_state['drivers'][:3]
                pos_str = ', '.join([f"{d.get('driver', '?')}:P{d.get('position', '?')}" for d in top_3])
                print(f"[LAP-DATA-DEBUG] Lap {lap_number}: Positions - {pos_str}")
            
            # ===== DETECT BATTLES FIRST - before other events =====
            # This ensures battle events are not overwritten by pit stop detection
            try:
                # Detect battles between drivers
                battle_events = self.battle_detector.detect_battles(lap_number, lap_state['drivers'])
                
                # Debug: show driver data for top 5
                if lap_number <= 3:
                    print(f"[DEBUG-GAPS] Lap {lap_number}: Top 5 drivers gap data:")
                    for driver in lap_state['drivers'][:5]:
                        print(f"   P{driver.get('position')}: {driver.get('driver')} - gap_to_leader={driver.get('gap_to_leader')}, gap_to_next={driver.get('gap_to_next')}")
                
                # Convert battle events to notification events
                for battle_event in battle_events:
                    event = self.event_generator.generate_battle_event(battle_event)
                    if event:
                        lap_state['events'].append(event)
                
                # Debug logging: show what happened
                if len(battle_events) > 0:
                    print(f"[EVENTS] Lap {lap_number}: Generated {len(battle_events)} battle event(s): {[e.get('subtype', 'unknown') for e in battle_events]}")
                else:
                    # Show active battles to debug
                    active_count = len(self.battle_detector.active_battles)
                    if active_count > 0:
                        print(f"[EVENTS] Lap {lap_number}: {active_count} active battle(s), no new events (no gap changes > threshold)")
                    else:
                        print(f"[EVENTS] Lap {lap_number}: No battles detected (gaps may be 0 or all > 0.8s)")
                        
            except Exception as e:
                print(f"[WARN] Battle detection error: {e}")
                import traceback
                traceback.print_exc()
            
            # THEN detect other events (pit stops, DNF, etc.)
            lap_state['events'].extend(self._detect_events(lap_number, lap_data))
        else:
            # Only simulate if no real data available
            print(f"[LAP {lap_number}] ⚠ NO FastF1 data available - using simulated fallback (this may cause unrealistic gaps!)")
            lap_state['drivers'] = self._simulate_lap_changes()
            lap_state['predictions'] = self._get_predictions(lap_number, None)
        
        # ===== DISABLED: Test events replaced with real battle/pit stop detection =====
        # Removed guaranteed test events - now using actual race event detection instead
        # Test events were only for verification; real events now trigger notifications
        
        # Add MODEL METRICS for frontend display
        if self.model is not None:
            try:
                # V3 model uses updates_count (not lap_updates_count)
                lap_state['model_metrics'] = {
                    'total_updates': getattr(self.model, 'updates_count', 0),
                    'model_maturity_percentage': min(100, getattr(self.model, 'updates_count', 0) * 5),
                    'position_model_ready': self.model.features_fitted,
                    'features_fitted': getattr(self.model, 'features_fitted', False)
                }
            except Exception as e:
                print(f"[WARN] Could not get model metrics: {e}")
                lap_state['model_metrics'] = {}
        
        return lap_state
    
    def _get_lap_data(self, lap_number):
        """Get real lap data from FastF1"""
        try:
            if self.laps_data is None or len(self.laps_data) == 0:
                if lap_number <= 3:
                    print(f"[GETLAPDATA] Lap {lap_number}: laps_data is empty/None!")
                return None
            
            lap_filter = self.laps_data[self.laps_data['LapNumber'] == lap_number]
            if len(lap_filter) == 0:
                if lap_number <= 5:
                    unique_laps = sorted(self.laps_data['LapNumber'].unique())
                    print(f"[GETLAPDATA] Lap {lap_number}: NO ROWS FOUND! Available laps: {unique_laps[:5]}")
            return lap_filter
        except Exception as e:
            if lap_number <= 3:
                print(f"[GETLAPDATA] Lap {lap_number}: Exception - {e}")
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
                
                # Sanity check: Prevent unrealistic jumps (e.g. P11 -> P3 in lap 1) due to data glitches
                # Only apply strict check on lap 1
                if lap_number == 1 and abs(state['position'] - new_position) > 8:
                    # If jump is too big in lap 1, trust the grid/current position more unless confirmed
                    pass # Keep existing position for now to avoid UI glitching
                
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
        
        # ===== CUMULATIVE GAP CALCULATION (realistic!) =====
        # Gap grows EACH LAP by the difference in lap times.
        # Example: If VER does 88.0s and NOR does 88.4s every lap:
        #   Lap 1: gap = +0.4s
        #   Lap 2: gap = +0.8s
        #   Lap 3: gap = +1.2s  ... etc.
        # This is how real F1 timing works.
        
        # Step 1: Find the leader's lap time for this lap
        leader_lap_time = None
        for driver in active_drivers:
            code = driver.get('driver')
            if code in self.driver_states and self.driver_states[code]['lap_times']:
                leader_lap_time = self.driver_states[code]['lap_times'][-1]
                break  # active_drivers is sorted by position, so first = leader
        
        # Step 2: For each driver, update cumulative gap based on this lap's time delta
        for i, driver in enumerate(active_drivers):
            code = driver.get('driver')
            if code not in self.driver_states:
                continue
            
            state = self.driver_states[code]
            
            if i == 0:
                # Leader: gap is always 0
                state['cumulative_gap'] = 0.0
                active_drivers[i]['gap_to_leader'] = 0.0
                active_drivers[i]['gap'] = '0.000'
            else:
                if leader_lap_time is not None and state['lap_times']:
                    this_lap_time = state['lap_times'][-1]
                    # Gap delta this lap = how much slower this driver was vs leader
                    lap_delta = this_lap_time - leader_lap_time
                    # Accumulate the gap (can grow OR shrink if driver is faster)
                    state['cumulative_gap'] = max(0.0, state['cumulative_gap'] + lap_delta)
                else:
                    # No lap time data: keep previous gap (don't reset to 0!)
                    pass
                
                active_drivers[i]['gap_to_leader'] = round(state['cumulative_gap'], 3)
                active_drivers[i]['gap'] = f"+{state['cumulative_gap']:.3f}"
        
        # Step 3: Compute gap_to_next (gap between driver[i] and driver[i-1])
        for i in range(1, len(active_drivers)):
            gap_i = active_drivers[i]['gap_to_leader']
            gap_prev = active_drivers[i - 1]['gap_to_leader']
            active_drivers[i]['gap_to_next'] = round(max(0.0, gap_i - gap_prev), 3)
        
        # Train AI model with this lap's data
        if self.model is not None:
            try:
                # Prepare lap data for training - must match pre-training features!
                # Pre-trained on: grid, grid_position, driver_age, points_constructor, circuitId, constructorId, year
                                # V3 Model: Use add_race_data() to add data with proper tire/pit stop tracking
                # Format data with all required fields for advanced pit stop/tire analysis
                lap_drivers_for_model = []
                for driver_code, state in self.driver_states.items():
                    if not state.get('dnf', False):  # Skip DNF drivers
                        lap_drivers_for_model.append({
                            'driver': driver_code,
                            'position': state['position'],
                            'lap_time': state['lap_times'][-1] if state['lap_times'] else 90.0,
                            'tire_compound': state['tire_compound'],
                            'tire_age': state['tire_age'],
                            'pit_stops': state['pit_stops'],
                            'grid_position': state['grid_position'],
                            'points_constructor': 100.0  # Default team strength
                        })
                
                # Add this lap's data to the model buffer
                self.model.add_race_data(lap_number, lap_drivers_for_model)
                
                # Train model EVERY LAP with continuous learning (v2 API)
                result = self.model.update_model(lap_number)
                if result.get('status') == 'updated':
                    print(f"[MODEL] Trained on {result.get('samples', 0)} drivers - LAP {lap_number}")
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
            'driver': driver_code,  # Key for battle detector - must be 'driver' not 'driver_code'
            'driver_code': driver_code,
            'driver_name': state['driver_name'],
            'team': state['team'],
            'lap_time': state['current_lap_time'] or '--:--',
            'tire_compound': state['tire_compound'],
            'tire_age': state['tire_age'],
            'pit_stops': state['pit_stops'],
            'gap': state.get('gap_to_leader', 0.0),  # Real gap data for battle detection
            'gap_to_next': state.get('gap_to_next', 0.0),
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
                # Use Advanced AI model predict_lap() for better predictions
                # Build drivers_data from current states
                drivers_data = []
                for driver_code, state in self.driver_states.items():
                    if not state.get('dnf', False):  # Skip DNF drivers
                        drivers_data.append({
                            'driver_code': driver_code,
                            'driver': driver_code,
                            'position': state['position'],
                            'grid_position': state['grid_position'],
                            'tire_age': state.get('tire_age', 1),
                            'tire_compound': state.get('tire_compound', 'MEDIUM'),
                            'pit_stops': state.get('pit_stops', 0),
                            'constructor': state.get('team', 'Unknown'),
                            'driver_number': state.get('driver_number', 0),
                            'driver_age': state.get('driver_age', 28),
                            'points_constructor': 100.0,
                            'lap_time': state.get('current_lap_time', 100.0),
                            'gap_to_leader': abs(state.get('gap_to_leader', 0.0)),
                            'gap': state.get('gap_to_driver_ahead', 0.0),
                            'current_lap': lap_number  # IMPORTANT: Pass lap number for race-phase confidence scaling
                        })
                
                # Call predict_lap() - v2 API uses only drivers_lap_data parameter
                if drivers_data and self.model:
                    model_predictions_dict = self.model.predict_lap(drivers_data)
                    
                    # Convert to output format
                    # v2 predict_lap returns Dict[str, Tuple[pred_pos, confidence]]
                    predictions = []
                    for driver_code, pred_tuple in model_predictions_dict.items():
                        state = self.driver_states.get(driver_code, {})
                        
                        # pred_tuple is (predicted_position, confidence)
                        pred_pos, confidence = pred_tuple if isinstance(pred_tuple, tuple) else (15, 50.0)
                        
                        predictions.append({
                            'position': state.get('position', 15),
                            'driver_code': driver_code,
                            'driver_name': state.get('driver_name', driver_code),
                            'confidence': float(confidence),
                            'prediction': int(pred_pos),
                            'trend': 'stable',  # v2 doesn't provide trend, use default
                            'grid_pos': state.get('grid_position', 15),
                            'pit_stops': state.get('pit_stops', 0)
                        })
                
                # SORT by confidence (winnerkans) and take TOP 5 ONLY
                predictions.sort(key=lambda x: x['confidence'], reverse=True)
                top_5_predictions = predictions[:5]
                
                # Log for debugging - show realistic top 5
                top5_strs = [f"{p['driver_code']}({p['confidence']:.0f}%)" for p in top_5_predictions]
                print(f"[PREDICTIONS] Lap {lap_number} - Top 5 Winners: {top5_strs}")
                
                # DEBUG: Log confidence for first 3 drivers
                if lap_number <= 5:
                    for p in predictions[:3]:
                        print(f"[TOP5-DEBUG] Lap {lap_number}: {p['driver_code']} confidence={p['confidence']:.1f}% (pos={p['position']}, pred={p['prediction']:.1f})")
                
                # Return only top 5
                predictions = top_5_predictions
                
        except Exception as e:
            print(f"[SIMULATOR] Error getting predictions: {str(e)}")
            import traceback
            traceback.print_exc()
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
            
            # ===== PIT STOP DETECTION =====
            # Track tire changes to detect pit stops
            for driver_data in lap_data:
                # Skip if not a dict (sometimes data might be malformed)
                if not isinstance(driver_data, dict):
                    continue
                
                driver_code = driver_data.get('driver', '?')
                current_tire = driver_data.get('tire_compound', 'UNKNOWN')
                current_pit_stops = driver_data.get('pit_stops', 0)
                
                # Skip if no valid driver code
                if not driver_code or driver_code == '?':
                    continue
                
                # Initialize driver tracking if needed
                if driver_code not in self.previous_driver_state:
                    self.previous_driver_state[driver_code] = {
                        'tire': current_tire,
                        'pit_stops': current_pit_stops
                    }
                
                prev_state = self.previous_driver_state[driver_code]
                prev_tire = prev_state.get('tire', 'UNKNOWN')
                prev_pit_stops = prev_state.get('pit_stops', 0)
                
                # Detect pit stop: tire changed AND pit stop counter increased
                if current_tire != prev_tire and current_pit_stops > prev_pit_stops:
                    pit_event = {
                        'id': int(__import__('time').time() * 1000),
                        'timestamp': __import__('datetime').datetime.now().isoformat(),
                        'type': 'pit_stop',
                        'subtype': 'pit_stop',
                        'message': f'🔧 {driver_code}: Pit stop - changed to {current_tire} (Stop #{current_pit_stops})',
                        'driver': driver_code,
                        'lap': lap_number,
                        'new_tire': current_tire,
                        'stop_number': current_pit_stops
                    }
                    events.append(pit_event)
                    print(f"[PIT-STOP] Lap {lap_number}: {driver_code} pit stop detected - new tire: {current_tire}")
                
                # Update state for next lap
                self.previous_driver_state[driver_code] = {
                    'tire': current_tire,
                    'pit_stops': current_pit_stops
                }
            
            return events
                
        except Exception as e:
            print(f"[WARN] Error detecting events: {e}")
            import traceback
            traceback.print_exc()
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
