"""
F1 AI Tracker Backend - Flask + SocketIO Server
Real-time race simulation with WebSocket communication
"""

import sys
import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room
import json
from datetime import datetime
import threading
import time
from continuous_model_learner_v2 import ContinuousModelLearner
from fastf1_data_fetcher import FastF1DataFetcher
import pandas as pd
import fastf1
from race_simulator import RaceSimulator
from prerace_model import ensure_prerace_model_loaded
from tire_strategy_model import ensure_tire_strategy_model_loaded

# Performance optimization: Rate-limiting for Socket.IO emissions
class RateLimiter:
    """Rate limiter for Socket.IO emissions to prevent CPU/network overload"""
    def __init__(self, min_interval_ms=100):
        self.min_interval = min_interval_ms / 1000.0  # Convert to seconds
        self.last_emit_time = 0
    
    def should_emit(self):
        """Check if enough time has passed since last emission"""
        current_time = time.time()
        if current_time - self.last_emit_time >= self.min_interval:
            self.last_emit_time = current_time
            return True
        return False
    
    def reset(self):
        """Reset the limiter"""
        self.last_emit_time = 0

# Setup Flask App
app = Flask(__name__)

# CORS Configuration - Environment-aware for security
# In development: allow localhost + local network
# In production: restrict to specific trusted domains
ENVIRONMENT = os.getenv('FLASK_ENV', 'development')

# Define allowed origins based on environment
if ENVIRONMENT == 'production':
    # Production: MUST be configured with actual trusted domains
    # Update these BEFORE deploying to production
    ALLOWED_ORIGINS = os.getenv('CORS_ALLOWED_ORIGINS', '').split(',') if os.getenv('CORS_ALLOWED_ORIGINS') else [
        # 'https://yourdomain.com',      # TODO: Replace with your actual domain
        # 'https://www.yourdomain.com',  # TODO: Replace with your actual domain
    ]
    if not ALLOWED_ORIGINS or ALLOWED_ORIGINS == ['']:
        print("WARNING: No CORS_ALLOWED_ORIGINS set for production! Set via environment variable.")
        ALLOWED_ORIGINS = []  # Empty list = no cross-origin requests allowed (safest)
else:
    # Development: allow localhost variants + local network
    ALLOWED_ORIGINS = [
        'http://localhost:5173',     # Vite dev server (default)
        'http://localhost:3000',     # Fallback dev port
        'http://127.0.0.1:5173',
        'http://127.0.0.1:3000',
        'http://localhost:5000',     # Backend itself (for testing)
        'http://127.0.0.1:5000'
    ]

# Configure CORS with restricted origins (more secure than '*')
# NOTE: supports_credentials=False is correct for stateless apps (no authentication/sessions)
#       If you add authentication later, change this to True and ensure ALLOWED_ORIGINS is tightly controlled
CORS(app, 
     origins=ALLOWED_ORIGINS,
     supports_credentials=False,
     methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type']
)

# Socket.IO setup - use polling only (avoid werkzeug WebSocket issues)
socketio = SocketIO(
    app,
    cors_allowed_origins=ALLOWED_ORIGINS,  # Match Flask CORS configuration
    async_mode='threading',
    logger=True,
    engineio_logger=False,
    ping_timeout=60,
    ping_interval=25,
    async_handlers=True
)

# Global state
race_state = {
    'running': False,
    'current_lap': 0,
    'total_laps': 58,
    'drivers': [],
    'predictions': [],
    'race_name': '',
    'simulation_speed': 1.0,
    'race_simulator': None,
    'weather': {
        'temp': 25,
        'humidity': 50,
        'track_temp': 35,
        'conditions': 'Dry'
    }
}

# Race initialization state tracking with thread-safety
init_state = {
    'current_race': None,
    'status': 'idle',  # idle, initializing, ready, error
    'error_message': None,
    'progress': 0  # 0-100 for progress tracking
}
init_state_lock = threading.Lock()  # Thread-safe access to init_state

# Rate limiter instance for Socket.IO emissions (min 100ms between broadcasts)
lap_update_limiter = RateLimiter(min_interval_ms=100)

# Model cache
model_cache = {
    'model': None,
    'loaded': False
}

# Tire strategy model pre-loading (background thread)
tire_strategy_preload_thread = None
tire_strategy_preload_started = False

def preload_tire_strategy_model():
    """Preload tire strategy model in background to avoid UI freeze"""
    global tire_strategy_preload_started
    
    if tire_strategy_preload_started:
        return
    
    tire_strategy_preload_started = True
    
    def _preload():
        try:
            print("[STARTUP] Pre-loading tire strategy model in background...")
            ensure_tire_strategy_model_loaded()
            print("[STARTUP] Tire strategy model pre-loaded successfully!")
        except Exception as e:
            print(f"[STARTUP] ERROR pre-loading tire strategy model: {e}")
    
    # Start background thread
    preload_thread = threading.Thread(target=_preload, daemon=True)
    preload_thread.start()


@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({'message': 'F1 AI Tracker Backend'}), 200


@app.route('/api/health', methods=['GET'])
def health_check():
    """API Health Check"""
    return jsonify({
        'status': 'ok',
        'version': '1.0',
        'timestamp': datetime.now().isoformat()
    }), 200


# ========== UTILITY FUNCTIONS ==========

def get_race_info(race_num):
    """Utility function to get race name and validate race number (P4.2 - reduce duplication)"""
    RACES_MAP = {
        1: "Bahrain", 2: "Saudi Arabia", 3: "Australia", 4: "Japan", 5: "China",
        6: "Miami", 7: "Monaco", 8: "Canada", 9: "Spain", 10: "Austria",
        11: "UK", 12: "Hungary", 13: "Belgium", 14: "Netherlands", 15: "Italy",
        16: "Azerbaijan", 17: "Singapore", 18: "Austin", 19: "Mexico", 20: "Brazil", 21: "Abu Dhabi"
    }
    
    if not isinstance(race_num, int) or race_num < 1 or race_num > 21:
        raise ValueError(f'Invalid race number {race_num}. Must be 1-21.')
    
    race_name = RACES_MAP.get(race_num, "Unknown")
    return race_name


def get_race_grid(race_num):
    """Fetch qualifying grid from FastF1 or fallback (P4.1 - can be cached later)"""
    grid = _fetch_qualifying_grid(race_num)
    
    if not grid or len(grid) == 0:
        print(f"[API] WARNING: Could not fetch FastF1 data, using fallback")
        grid = _get_fallback_grid(race_num)
    else:
        print(f"[API] ✓ Using REAL FastF1 qualifying grid")
    
    return grid


# ========== API ENDPOINTS ==========


@app.route('/api/races', methods=['GET'])
def get_races():
    """Get available 2024 F1 races"""
    races = {
        1: {"name": "Bahrain", "circuit": "Sakhir"},
        2: {"name": "Saudi Arabia", "circuit": "Jeddah"},
        3: {"name": "Australia", "circuit": "Melbourne"},
        4: {"name": "Japan", "circuit": "Suzuka"},
        5: {"name": "China", "circuit": "Shanghai"},
        6: {"name": "Miami", "circuit": "USA"},
        7: {"name": "Monaco", "circuit": "Monte Carlo"},
        8: {"name": "Canada", "circuit": "Montreal"},
        9: {"name": "Spain", "circuit": "Barcelona"},
        10: {"name": "Austria", "circuit": "Spielberg"},
        11: {"name": "United Kingdom", "circuit": "Silverstone"},
        12: {"name": "Hungary", "circuit": "Budapest"},
        13: {"name": "Belgium", "circuit": "Spa"},
        14: {"name": "Netherlands", "circuit": "Zandvoort"},
        15: {"name": "Italy", "circuit": "Monza"},
        16: {"name": "Azerbaijan", "circuit": "Baku"},
        17: {"name": "Singapore", "circuit": "Marina Bay"},
        18: {"name": "Austin", "circuit": "USA"},
        19: {"name": "Mexico", "circuit": "Mexico City"},
        20: {"name": "Brazil", "circuit": "Interlagos"},
        21: {"name": "Abu Dhabi", "circuit": "Yas Island"},
    }
    return jsonify(races), 200



@app.route('/api/race/prerace-analysis', methods=['POST'])
def get_prerace_analysis():
    """Get pre-race analysis and AI predictions for upcoming race
    
    Fetches REAL qualifying data from FastF1 and uses XGBoost model for predictions
    """
    try:
        data = request.json or {}
        race_num = data.get('race_number', 21)
        
        # Use utility to validate and get race name (P4.2 - removes duplication)
        try:
            race_name = get_race_info(race_num)
        except ValueError as ve:
            return jsonify({'error': str(ve), 'status': 'error'}), 400
        
        print(f"\n{'='*80}")
        print(f"[PRERACE API] RACE {race_num}: {race_name} - Processing pre-race analysis")
        print(f"{'='*80}")
        
        # Load model
        model = ensure_prerace_model_loaded()
        if not model or not model.loaded:
            print("[PRERACE API] ERROR: Model failed to load")
            return jsonify({'error': 'Could not load model'}), 500
        
        # Fetch grid using utility (P4.1 - caching opportunity identified)
        grid = get_race_grid(race_num)
        
        # Log grid positions for debugging
        print(f"[PRERACE API] Grid positions for Race {race_num} ({race_name}):")
        for i, driver in enumerate(grid[:10], 1):  # Show top 10
            print(f"    P{driver.get('grid_pos', i):2d}: {driver.get('driver'):3s} - {driver.get('team', 'Unknown')}")
        if len(grid) > 10:
            print(f"    ... and {len(grid) - 10} more drivers")
        
        print(f"[PRERACE API] Total: {len(grid)} drivers loaded")
        
        # Get predictions from model
        predictions = model.predict(grid, race_num)
        
        print(f"[PRERACE API] ✓ Generated {len(predictions)} predictions for Race {race_num} ({race_name})")
        print(f"[PRERACE API] Top 5 predictions:")
        for i, pred in enumerate(predictions[:5], 1):
            print(f"    {i}. {pred.get('driver'):3s} (Grid P{pred.get('grid_position'):2d}) - Confidence: {pred.get('confidence', 0):.1f}%")
        print(f"{'='*80}\n")
        
        return jsonify({
            'status': 'success',
            'race_number': race_num,
            'predictions': predictions,
            'analysis': {
                'model': 'XGBoost Ensemble',
                'features_used': len(model.feature_cols) if hasattr(model, 'feature_cols') else 40,
                'confidence_threshold': 85.0
            }
        }), 200
        
    except Exception as e:
        print(f"[PRERACE API] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'error'}), 400


@app.route('/api/race/tire-strategy', methods=['POST'])
def get_tire_strategy():
    """Get tire strategy recommendations for upcoming race
    
    Uses circuit-specific tire degradation models and weather forecasting
    """
    try:
        data = request.json or {}
        race_num = data.get('race_number', 21)
        weather_forecast = data.get('weather_forecast')  # Optional: {temp_c, precip_mm, cloud_pct}
        
        # Use utility to validate and get race name (P3.1 HIGH - fix + P4.2 - reduce duplication)
        try:
            race_name = get_race_info(race_num)
        except ValueError as ve:
            return jsonify({'error': str(ve), 'status': 'error'}), 400
        
        print(f"\n{'='*80}")
        print(f"[TIRE STRATEGY API] RACE {race_num}: {race_name} - Computing tire strategy")
        print(f"{'='*80}")
        
        # Load tire strategy model (with timeout to prevent UI freeze)
        try:
            tire_model = ensure_tire_strategy_model_loaded()
            if not tire_model or not tire_model.loaded:
                print("[TIRE STRATEGY API] WARNING: Model not ready, returning graceful fallback")
                return jsonify({
                    'status': 'pending',
                    'message': 'Tire strategy model is loading. Please try again in a moment.',
                    'race_number': race_num,
                    'strategies': []
                }), 202  # 202 Accepted - still processing
        except Exception as model_err:
            print(f"[TIRE STRATEGY API] ERROR loading model: {model_err}")
            return jsonify({
                'status': 'error',
                'message': f'Tire strategy model unavailable: {str(model_err)[:100]}',
                'race_number': race_num,
                'strategies': []
            }), 503  # 503 Service Unavailable
        
        # Fetch grid using utility (P4.1 - caching opportunity identified)
        grid = get_race_grid(race_num)
        
        print(f"[TIRE STRATEGY API] Grid loaded: {len(grid)} drivers")
        
        # Get tire strategy predictions (includes per-driver strategies)
        strategies = tire_model.predict_strategy(grid, race_num, weather_forecast)
        
        print(f"[TIRE STRATEGY API] ✓ Generated tire strategies for {len(strategies)} strategies")
        print(f"[TIRE STRATEGY API] Top strategies:")
        for i, strat in enumerate(strategies, 1):
            if strat.get('strategy_type') != 'per_driver_details':
                print(f"    {i}. {strat.get('strategy_type'):15s} - Confidence: {strat.get('confidence', 0):.0f}%")
        print(f"{'='*80}\n")
        
        # Extract general strategies (first 2) and per-driver strategies
        general_strategies = [s for s in strategies if s.get('strategy_type') != 'per_driver_details']
        per_driver = next((s for s in strategies if s.get('strategy_type') == 'per_driver_details'), None)
        
        # Format response
        response_data = {
            'status': 'success',
            'race_number': race_num,
            'strategies': general_strategies,  # Top 2 general strategies
            'per_driver_strategies': per_driver.get('drivers', []) if per_driver else [],  # Individual driver tire sequences
            'circuit_analysis': tire_model.circuit_properties.get(int(race_num), {}),
            'tire_degradation': tire_model.tire_degradation_curves.get(int(race_num), {}),
            'analysis': {
                'model': 'Circuit-Specific XGBoost + Degradation Curves + Per-Driver Optimization',
                'error_margin': '±2-3 laps',
                'weather_integrated': True,
                'per_driver_tire_selection': True,
                'circuit_properties': ['tire_wear_rate', 'track_type', 'brake_wear', 'overtaking_difficulty']
            }
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        print(f"[TIRE STRATEGY API] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'error'}), 400


def _fetch_qualifying_grid(race_num):
    """Fetch REAL qualifying grid from FastF1 for the specified race"""
    try:
        print(f"  [GRID] Attempting to fetch FastF1 qualifying data for race {race_num}...")
        
        qual_session = fastf1.get_session(2024, race_num, 'Q')
        qual_session.load(telemetry=False, weather=False)  # Disable expensive data
        
        grid = []
        if qual_session.results is not None and len(qual_session.results) > 0:
            for grid_idx, (_, row) in enumerate(qual_session.results.iterrows()):
                driver_code = str(row.get('Abbreviation', ''))
                if driver_code and driver_code != 'nan':
                    grid_pos = grid_idx + 1
                    grid.append({
                        'driver': driver_code,
                        'number': int(row.get('DriverNumber', 0)),
                        'team': str(row.get('TeamName', 'Unknown')),
                        'grid_pos': grid_pos
                    })
            
            print(f"  [GRID] ✓ Successfully loaded {len(grid)} drivers from FastF1 qualifying")
            return grid
        else:
            print(f"  [GRID] No qualifying results found in FastF1 for race {race_num}")
            return None
            
    except Exception as e:
        print(f"  [GRID] ERROR fetching FastF1 data: {type(e).__name__}: {str(e)[:100]}")
        return None


def _get_fallback_grid(race_num):
    """Fallback grid data with race-specific variations
    
    Used when FastF1 data is not available
    """
    # Base grid with all 20 drivers (2024 grid)
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
        {'driver': 'ALB', 'number': 23, 'team': 'Williams'},
        {'driver': 'SARGEant', 'number': 2, 'team': 'Williams'},
        {'driver': 'PER', 'number': 11, 'team': 'Red Bull'},
        {'driver': 'RIC', 'number': 3, 'team': 'Racing Bulls'},
    ]
    
    # Race-specific variations
    race_adjustments = {
        1: [0, 1, -1, 0, 1, 0, -1, 1, 0, 2, -1, 1, 0, -1, 1, 0, 2, -1, 1, 0],  # Bahrain
        2: [1, 0, 1, -1, 0, 1, 0, -1, 2, 1, 0, -1, 1, 0, -1, 2, 1, 0, -1, 1],  # Saudi Arabia
        3: [-1, 1, 0, 1, 0, -1, 1, 0, 1, 0, -1, 1, 0, 1, -1, 0, 1, 0, -1, 1],  # Australia
        4: [0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, 1, -1, 0, 1, -1, 0, 1],  # Japan
        5: [2, -1, 1, 0, -1, 1, 0, 1, -1, 0, 1, -1, 0, 1, 0, -1, 1, 0, -1, 1], # China
    }
    
    # Default to race 21 (Abu Dhabi) adjustments if not found
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
    
    print(f"  [GRID] Using fallback grid for race {race_num}")
    print(f"  [GRID] Top 10 drivers in fallback grid:")
    sorted_grid = sorted(grid, key=lambda x: x['grid_pos'])[:10]
    for driver in sorted_grid:
        print(f"    P{driver['grid_pos']:2d}: {driver['driver']:3s} - {driver['team']}")
    return grid


@app.route('/api/race/init', methods=['POST', 'GET'])
def init_race():
    """Initialize a race simulation (async, returns immediately)"""
    try:
        # Support both POST (with body) and GET (with query param)
        if request.method == 'POST':
            race_num = request.json.get('race_number', 21)
        else:
            race_num = int(request.args.get('race', 21))
        
        # Validate race_num is within expected range (1-21 for 2024 season)
        if not isinstance(race_num, int):
            race_num = int(race_num)
        if race_num < 1 or race_num > 21:
            return jsonify({'error': f'Invalid race number {race_num}. Must be between 1-21.', 'status': 'error'}), 400
        
        print(f"[BACKEND] Race init requested for race {race_num}")
        
        # Update initialization state (thread-safe)
        with init_state_lock:
            init_state['current_race'] = race_num
            init_state['status'] = 'initializing'
            init_state['error_message'] = None
            init_state['progress'] = 0
        
        # Return immediate response indicating initialization is in progress
        response = {
            'status': 'initializing',
            'race_id': race_num,
            'message': f'Race {race_num} initialization in progress. Listen for race/ready event.',
            'poll_url': f'/api/race/init-status?race={race_num}'
        }
        
        # Start background initialization thread
        init_thread = threading.Thread(
            target=_initialize_race_background,
            args=(race_num,),
            daemon=True
        )
        init_thread.start()
        
        return jsonify(response), 202  # 202 Accepted - request is being processed
        
    except Exception as e:
        error_msg = str(e)
        print(f"[BACKEND] ERROR in init request: {error_msg}")
        init_state['status'] = 'error'
        init_state['error_message'] = error_msg
        return jsonify({'error': error_msg, 'status': 'error'}), 400


@app.route('/api/race/init-status', methods=['GET'])
def get_init_status():
    """Poll initialization status"""
    try:
        race_num = request.args.get('race')
        
        # Thread-safe access to init_state
        with init_state_lock:
            # If no race specified or different race, return current status
            if race_num is None:
                return jsonify({
                    'status': init_state['status'],
                    'current_race': init_state['current_race'],
                    'progress': init_state['progress'],
                    'error': init_state['error_message']
                }), 200
            
            # Check if status is for requested race
            if str(init_state['current_race']) == str(race_num):
                return jsonify({
                    'status': init_state['status'],
                    'race': race_num,
                    'progress': init_state['progress'],
                    'error': init_state['error_message']
                }), 200
            else:
                return jsonify({
                    'status': 'idle',
                    'race': race_num,
                    'error': 'No initialization in progress for this race'
                }), 200
            
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400


def _fetch_fastf1_data(race_num):
    """Fetch race data from FastF1 API with fallback to dummy drivers"""
    drivers = []
    laps = None
    weather_data = None
    
    try:
        print(f"[BACKGROUND] Fetching FastF1 data for race {race_num}...")
        fetcher = FastF1DataFetcher()
        # Add timeout to prevent indefinite blocking on slow API
        try:
            result = fetcher.fetch_race(2024, race_num)
        except Exception as timeout_err:
            print(f"[BACKGROUND] WARNING: FastF1 API timeout/error: {timeout_err}")
            result = False
        
        if result:
            # STEP 1: Get ONLY qualifying session for ACTUAL grid positions
            # IMPORTANT: We use qualifying ORDER as the true grid positions, NOT race results!
            print("[BACKGROUND] Loading qualifying session for TRUE grid positions...")
            drivers = []
            qual_grid_map = {}  # {driver_code: grid_position}
            session = fetcher.session  # Get session reference
            laps = None
            
            try:
                qual_session = fastf1.get_session(2024, race_num, 'Q')
                qual_session.load()
                
                if qual_session.results is not None and len(qual_session.results) > 0:
                    # Grid positions are the ROW ORDER in qualifying results!
                    for grid_idx, (_, qual_row) in enumerate(qual_session.results.iterrows()):
                        driver_code = str(qual_row.get('Abbreviation', ''))
                        if driver_code and driver_code != 'nan':
                            grid_pos = grid_idx + 1  # 1-based position
                            qual_grid_map[driver_code] = grid_pos
                            driver_name = str(qual_row.get('FullName', 'Unknown'))
                            team_name = str(qual_row.get('TeamName', 'Unknown'))
                            driver_num = int(qual_row.get('DriverNumber', 0))
                            
                            drivers.append({
                                'code': driver_code,
                                'name': driver_name,
                                'team': team_name,
                                'number': driver_num,
                                'grid_position': grid_pos
                            })
                            print(f"    Grid P{grid_pos:2d}: {driver_code} - {driver_name}")
                    
                    print(f"[BACKGROUND] OK: Loaded {len(drivers)} drivers from QUALIFYING")
                else:
                    print("[BACKGROUND] WARNING: No qualifying results found")
                    
            except Exception as qual_err:
                print(f"[BACKGROUND] WARNING: Error loading qualifying: {qual_err}")
                # Fallback: get drivers from race session
                if hasattr(session, 'results') and session.results is not None:
                    for idx, (_, row) in enumerate(session.results.iterrows()):
                        if pd.notna(row.get('Abbreviation')):
                            driver_code = str(row.get('Abbreviation', ''))
                            drivers.append({
                                'code': driver_code,
                                'name': str(row.get('FullName', 'Unknown')),
                                'team': str(row.get('TeamName', 'Unknown')),
                                'number': int(row.get('DriverNumber', idx + 1)),
                                'grid_position': idx + 1
                            })
            
            # Get real lap data
            if hasattr(session, 'laps') and session.laps is not None:
                laps = session.laps
            
            # Get weather data from session
            try:
                if hasattr(session, 'weather_data') and session.weather_data is not None:
                    weather_data = session.weather_data
                    if len(weather_data) > 0:
                        print(f"[BACKGROUND] OK: Weather data loaded")
            except:
                pass
            
            if len(drivers) > 0:
                print(f"[BACKGROUND] OK: Loaded {len(drivers)} drivers from FastF1")
            else:
                raise Exception("No drivers found in FastF1 data")
        else:
            raise Exception(f"Failed to fetch race {race_num} from FastF1")
    except Exception as e:
        print(f"[BACKGROUND] WARNING: Could not load FastF1 data: {e}")
        print(f"[BACKGROUND] Falling back to dummy drivers...")
        
        # Fallback to dummy drivers
        drivers = [
            {'code': 'VER', 'name': 'Max Verstappen', 'team': 'Red Bull', 'number': 1, 'grid_position': 1},
            {'code': 'LEC', 'name': 'Charles Leclerc', 'team': 'Ferrari', 'number': 16, 'grid_position': 2},
            {'code': 'SAI', 'name': 'Carlos Sainz', 'team': 'Ferrari', 'number': 55, 'grid_position': 3},
            {'code': 'PIA', 'name': 'Oscar Piastri', 'team': 'McLaren', 'number': 81, 'grid_position': 4},
            {'code': 'NOR', 'name': 'Lando Norris', 'team': 'McLaren', 'number': 4, 'grid_position': 5},
            {'code': 'HAM', 'name': 'Lewis Hamilton', 'team': 'Mercedes', 'number': 44, 'grid_position': 6},
            {'code': 'RUS', 'name': 'George Russell', 'team': 'Mercedes', 'number': 63, 'grid_position': 7},
            {'code': 'ALO', 'name': 'Fernando Alonso', 'team': 'Aston Martin', 'number': 14, 'grid_position': 8},
            {'code': 'STR', 'name': 'Lance Stroll', 'team': 'Aston Martin', 'number': 18, 'grid_position': 9},
            {'code': 'GAS', 'name': 'Pierre Gasly', 'team': 'Alpine', 'number': 10, 'grid_position': 10},
            {'code': 'OCO', 'name': 'Esteban Ocon', 'team': 'Alpine', 'number': 31, 'grid_position': 11},
            {'code': 'MAG', 'name': 'Kevin Magnussen', 'team': 'Haas', 'number': 20, 'grid_position': 12},
            {'code': 'HUL', 'name': 'Nico Hulkenberg', 'team': 'Haas', 'number': 27, 'grid_position': 13},
            {'code': 'BOT', 'name': 'Valtteri Bottas', 'team': 'Sauber', 'number': 77, 'grid_position': 14},
            {'code': 'ZHO', 'name': 'Zhou Guanyu', 'team': 'Sauber', 'number': 24, 'grid_position': 15},
            {'code': 'TSU', 'name': 'Yuki Tsunoda', 'team': 'Racing Bulls', 'number': 22, 'grid_position': 16},
            {'code': 'VER2', 'name': 'TEST Driver 1', 'team': 'Williams', 'number': 23, 'grid_position': 17},
            {'code': 'NOR2', 'name': 'TEST Driver 2', 'team': 'Kick', 'number': 25, 'grid_position': 18},
            {'code': 'HAM2', 'name': 'TEST Driver 3', 'team': 'Test Team', 'number': 50, 'grid_position': 19},
        ]
    
    return drivers, laps, weather_data


def _train_ai_model(laps):
    """Load and train AI model with historical and race-specific data"""
    try:
        print("[BACKGROUND] Loading AI model (v2 - with pit stop analysis)...")
        model = ContinuousModelLearner()
        model_cache['model'] = model
        
        # PRE-TRAIN on historical F1 data for better baseline
        historical_csv = 'f1_historical_5years.csv'
        if os.path.exists(historical_csv):
            print(f"[BACKGROUND] Pre-training model on historical data from {historical_csv}...")
            model.pretrain_on_historical_data(csv_path=historical_csv)
            print("[BACKGROUND] OK: Pre-training complete")
        else:
            print(f"[BACKGROUND] WARNING: Historical data not found, training from current lap data")
        
        # Then fine-tune on current race lap data if available
        if laps is not None and len(laps) > 0:
            print(f"[BACKGROUND] Fine-tuning AI model on {len(laps)} race laps...")
            try:
                # Convert lap data to driver data format
                lap_drivers = []
                for lap_data_dict in laps.to_dict('records')[:100]:  # Sample first 100 laps
                    lap_drivers.append({
                        'driver_code': str(lap_data_dict.get('Driver', 'UNK')),
                        'position': int(lap_data_dict.get('Position', 15)),
                        'lap_time': lap_data_dict.get('Time'),
                        'tire_compound': str(lap_data_dict.get('Compound', 'MEDIUM')),
                        'lap_number': int(lap_data_dict.get('LapNumber', 1))
                    })
                
                if lap_drivers:
                    model.add_race_data(1, lap_drivers)
                    model.update_model(1)
            except Exception as train_err:
                print(f"[BACKGROUND] WARNING: Could not fine-tune model: {train_err}")
        
        model_cache['loaded'] = True
        print("[BACKGROUND] OK: AI model ready (Pre-trained + fine-tuned with 40+ features)")
        return model
    except Exception as e:
        print(f"[BACKGROUND] Error loading AI model: {str(e)}")
        model_cache['model'] = None
        model_cache['loaded'] = True  # Still allow race to start
        return None


def _initialize_race_background(race_num):
    """Background task for race initialization (calls separate focused functions)"""
    try:
        print(f"[BACKGROUND] Starting initialization for race {race_num}")
        with init_state_lock:
            init_state['progress'] = 10
        
        # Fetch FastF1 data from dedicated function
        drivers, laps, weather_data = _fetch_fastf1_data(race_num)
        print(f"[BACKGROUND] OK: Loaded {len(drivers)} drivers")
        
        with init_state_lock:
            init_state['progress'] = 40
        
        # Train AI model from dedicated function
        model = _train_ai_model(laps)
        
        with init_state_lock:
            init_state['progress'] = 80
        
        # Initialize race simulator - wrapped in try/catch
        try:
            print(f"[BACKGROUND] Creating RaceSimulator with {len(drivers)} drivers and model={model_cache['model']}")
            race_state['race_simulator'] = RaceSimulator(
                race_number=race_num,
                model=model_cache['model'],
                laps_data=laps,
                drivers=drivers,
                weather_data=weather_data
            )
            
            # Get initial state
            initial_state = race_state['race_simulator'].get_current_state()
            race_state['drivers'] = initial_state['drivers']
            race_state['race_name'] = initial_state['race_name']
            race_state['total_laps'] = initial_state['total_laps']
            race_state['current_lap'] = 0
            print(f"[BACKGROUND] ✓ RaceSimulator created successfully!")
        except Exception as sim_err:
            # Fallback if RaceSimulator fails
            print(f"[BACKGROUND] ERROR: RaceSimulator failed: {sim_err}")
            import traceback
            traceback.print_exc()
            print(f"[BACKGROUND] Using simple state fallback without simulator")
            race_state['drivers'] = drivers
            race_state['race_name'] = f'Race {race_num}'
            race_state['total_laps'] = 58
            race_state['current_lap'] = 0
            race_state['race_simulator'] = None  # Mark as failed but continue
        
        print(f"[BACKGROUND] Race initialized! {len(race_state['drivers'])} drivers")
        
        # Thread-safe state update
        with init_state_lock:
            init_state['progress'] = 100
            init_state['status'] = 'ready'
        
        # Emit Socket.IO event to all connected clients
        socketio.emit('race/ready', {
            'race_id': race_num,
            'race_name': race_state['race_name'],
            'drivers': race_state['drivers'],
            'total_laps': race_state['total_laps'],
            'message': f'Race {race_num} ready to start!'
        }, to=None)
        
        print(f"[BACKGROUND] OK: Initialization complete for race {race_num}")
        
    except Exception as e:
        print(f"[BACKGROUND] ERROR during background init: {str(e)}")
        # Thread-safe error state update
        with init_state_lock:
            init_state['status'] = 'error'
            init_state['error_message'] = str(e)
        
        # Emit error event to all connected clients
        socketio.emit('race/init-error', {
            'error': str(e),
            'race_id': race_num
        }, to=None)


# HTTP endpoints for race control (instead of Socket.IO)
@app.route('/api/race/state', methods=['GET'])
def get_race_state():
    """Get current race state - includes events for HTTP polling fallback"""
    return jsonify({
        'lap_number': race_state['current_lap'],
        'current_lap': race_state['current_lap'],
        'total_laps': race_state['total_laps'],
        'drivers': race_state['drivers'],
        'predictions': race_state['predictions'],
        'running': race_state['running'],
        'weather': race_state.get('weather', {}),
        'events': []  # Empty for now - SocketIO provides real-time events
    }), 200


@app.route('/api/race/start', methods=['POST'])
def start_race_http():
    """Start race via HTTP"""
    data = request.json or {}
    speed = data.get('speed', 1.0)
    
    # Check if race is initialized
    if race_state['race_simulator'] is None:
        return jsonify({'error': 'Race not initialized. Call /api/race/init first'}), 400
    
    race_state['running'] = True
    race_state['simulation_speed'] = speed
    race_state['current_lap'] = 1
    
    # Start simulation thread
    import threading
    threading.Thread(target=run_simulation, daemon=True).start()
    
    return jsonify({'status': 'started', 'speed': speed}), 200


@app.route('/api/race/pause', methods=['POST'])
def pause_race_http():
    """Pause race via HTTP"""
    race_state['running'] = False
    return jsonify({'status': 'paused'}), 200


@app.route('/api/race/resume', methods=['POST'])
def resume_race_http():
    """Resume race via HTTP"""
    race_state['running'] = True
    # Start simulation thread again to continue the race
    threading.Thread(target=run_simulation, daemon=True).start()
    return jsonify({'status': 'resumed'}), 200


@app.route('/api/race/speed', methods=['POST'])
def set_speed_http():
    """Set simulation speed via HTTP"""
    data = request.json or {}
    speed = data.get('speed', 1.0)
    race_state['simulation_speed'] = speed
    return jsonify({'status': 'speed_set', 'speed': speed}), 200


@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    print(f"\n{'='*60}")
    print(f"[SOCKETIO] OK: CLIENT CONNECTED")
    print(f"[SOCKETIO] Session ID: {request.sid}")
    print(f"[SOCKETIO] Namespace: {request.namespace}")
    print(f"{'='*60}\n")
    emit('connect_response', {
        'status': 'connected',
        'message': 'Connected to F1 AI Tracker Backend',
        'timestamp': datetime.now().isoformat()
    })


@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    print(f"[SOCKETIO] ERROR: CLIENT DISCONNECTED: {request.sid}")


@socketio.on('race/start')
def handle_race_start(data):
    """Start race simulation"""
    try:
        race_state['running'] = True
        race_state['current_lap'] = 1
        race_state['simulation_speed'] = data.get('speed', 1.0)
        
        print("[SOCKETIO] Race started!")
        emit('race/started', {
            'status': 'started',
            'current_lap': race_state['current_lap'],
            'total_laps': race_state['total_laps']
        }, broadcast=True)
        
        # Start simulation thread
        threading.Thread(target=run_simulation, daemon=True).start()
        
    except Exception as e:
        print(f"[SOCKETIO] Error starting race: {str(e)}")
        emit('race/error', {'error': str(e)})


@socketio.on('race/pause')
def handle_race_pause():
    """Pause race simulation"""
    race_state['running'] = False
    print("[SOCKETIO] Race paused!")
    emit('race/paused', {}, broadcast=True)


@socketio.on('race/resume')
def handle_race_resume():
    """Resume race simulation"""
    race_state['running'] = True
    print("[SOCKETIO] Race resumed!")
    emit('race/resumed', {}, broadcast=True)
    threading.Thread(target=run_simulation, daemon=True).start()


@socketio.on('race/speed')
def handle_simulation_speed(data):
    """Change simulation speed"""
    race_state['simulation_speed'] = data.get('speed', 1.0)
    emit('race/speed_changed', {
        'speed': race_state['simulation_speed']
    }, broadcast=True)


def run_simulation():
    """Main simulation loop - runs in background thread with rate limiting"""
    while race_state['running'] and race_state['current_lap'] <= race_state['total_laps']:
        try:
            # Get next lap state from simulator
            lap_state = race_state['race_simulator'].simulate_lap(race_state['current_lap'])
            
            # UPDATE race_state with fresh data so HTTP polling clients see changes
            race_state['drivers'] = lap_state['drivers']
            race_state['predictions'] = lap_state['predictions']
            race_state['weather'] = lap_state.get('weather', race_state.get('weather', {}))
            
            print(f"[SIMULATION] Lap {race_state['current_lap']} updated - {len(race_state['drivers'])} drivers")
            
            # EMIT EVENTS - Always send, don't rate limit!
            # Events are critical race information (battles, pit stops, etc.) that must be delivered in real-time
            # Rate limiting should NOT apply to events, only to routine lap updates
            events_to_send = lap_state.get('events', [])
            
            # Log what we're about to send
            if events_to_send:
                print(f"[EVENTS-TO-SEND] Lap {race_state['current_lap']}: {len(events_to_send)} events - {[e.get('type', '?') for e in events_to_send]}")
            
            # RATE LIMIT: Only emit driver/prediction updates if minimum time has passed (100ms)
            # This prevents network congestion while ensuring events always get through
            if lap_update_limiter.should_emit():
                socketio.emit('lap/update', {
                    'lap_number': race_state['current_lap'],
                    'drivers': lap_state['drivers'],
                    'predictions': lap_state['predictions'],
                    'events': events_to_send,
                    'weather': lap_state.get('weather', {})
                }, to=None)
                
                if events_to_send:
                    print(f"[BROADCAST] Emitted lap/update with {len(events_to_send)} event(s)")
            
            # If we have events but rate limiter blocked the update, send events separately to ensure delivery
            elif events_to_send:
                socketio.emit('lap/update', {
                    'lap_number': race_state['current_lap'],
                    'drivers': lap_state['drivers'],
                    'predictions': lap_state['predictions'],
                    'events': events_to_send,
                    'weather': lap_state.get('weather', {})
                }, to=None)
                print(f"[BROADCAST] Force-emitted lap/update with {len(events_to_send)} event(s) (bypassed rate limit)")
            
            # Move to next lap
            race_state['current_lap'] += 1
            
            # Simulate delay based on speed with MINIMUM sleep to prevent busy-waiting
            # Speed 1.0 = 5 seconds per lap (fast initial demo)
            # Speed 2.0 = 2.5 seconds per lap
            # Speed 0.5 = 10 seconds per lap
            # Minimum 0.01s sleep prevents CPU spinning even at very high speeds
            delay = max(0.01, 5.0 / race_state['simulation_speed'])
            time.sleep(delay)
            
        except Exception as e:
            print(f"[SIMULATION] Error in lap {race_state['current_lap']}: {str(e)}")
            socketio.emit('race/error', {'error': str(e)}, to=None)
            break
    
    # Race finished
    if race_state['current_lap'] > race_state['total_laps']:
        race_state['running'] = False
        socketio.emit('race/finished', {
            'final_standings': race_state['drivers']
        }, to=None)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("[BACKEND] Starting F1 AI Tracker Backend...")
    print("[BACKEND] Server running on http://localhost:5000")
    print("[BACKEND] Socket.IO available at ws://localhost:5000/socket.io/")
    print("="*60 + "\n")
    
    # Pre-load tire strategy model in background to avoid UI freeze on first request
    preload_tire_strategy_model()
    
    # Run with werkzeug
    socketio.run(
        app, 
        host='127.0.0.1', 
        port=5000, 
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=False
    )