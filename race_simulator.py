"""
Race Simulator - Simulates F1 race lap-by-lap with AI predictions
"""

import pandas as pd
import numpy as np
from copy import deepcopy
from datetime import timedelta


class RaceSimulator:
    """Simulates an F1 race with lap-by-lap updates and AI predictions"""
    
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
            'weather': self._get_weather_for_lap(lap_number)
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
        """Update driver states from real FastF1 data"""
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
        """Get AI predictions for next position"""
        predictions = []
        
        try:
            # Call model to get predictions
            if lap_data is not None and len(lap_data) > 0:
                lap_data_list = lap_data.to_dict('records')
            else:
                lap_data_list = list(self.driver_states.values())
            
            # For now, return confidence based on current position
            for driver_code, state in sorted(
                self.driver_states.items(),
                key=lambda x: x[1]['position']
            ):
                # Simple prediction: confidence decreases with position
                confidence = max(50, 100 - (state['position'] * 3))
                
                predictions.append({
                    'position': state['position'],
                    'driver_code': driver_code,
                    'confidence': confidence,
                    'trend': 'stable'  # Could be 'up', 'down', 'stable'
                })
        except Exception as e:
            print(f"[SIMULATOR] Error getting predictions: {str(e)}")
        
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
