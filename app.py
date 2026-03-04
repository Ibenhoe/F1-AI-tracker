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
from model.prerace_model import ensure_prerace_model_loaded
from model.tire_strategy_model import ensure_tire_strategy_model_loaded

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
        16: "Azerbaijan", 17: "Singapore", 18: "Austin", 19: "Mexico", 20: "Brazil",
        21: "Abu Dhabi", 22: "Las Vegas", 23: "Qatar", 24: "Abu Dhabi"
    }
    
    if not isinstance(race_num, int) or race_num < 1 or race_num > 24:
        raise ValueError(f'Invalid race number {race_num}. Must be 1-24.')
    
    race_name = RACES_MAP.get(race_num, "Unknown")
    return race_name


def get_race_grid(race_num):
    """Fetch qualifying grid from FastF1 or fallback (P4.1 - can be cached later)"""
    grid = _fetch_qualifying_grid(race_num)
    
    if not grid or len(grid) == 0:
        print(f"[API] WARNING: Could not fetch FastF1 data, using fallback")
        grid = _get_fallback_grid(race_num)
    # else:
    #     print(f"[API] ✓ Using REAL FastF1 qualifying grid")
    
    return grid

# ========== WIKI API ENDPOINTS (ADDED) ==========

def load_csv_data(filename):
    """Probeert CSV te laden uit de huidige map OF de F1_data_mangement map"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 0. Check in data/ map (NIEUW)
    path0 = os.path.join(current_dir, 'data', filename)
    if os.path.exists(path0):
        try:
            return pd.read_csv(path0).replace(r'\\N', None, regex=True)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            return pd.DataFrame()

    # 1. Check in F1-AI-tracker map (waar app.py staat)
    path1 = os.path.join(current_dir, filename)
    if os.path.exists(path1):
        try:
            return pd.read_csv(path1).replace(r'\\N', None, regex=True)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            return pd.DataFrame()

    # 2. Check in F1_data_mangement map (één map omhoog en dan in F1_data_mangement)
    path2 = os.path.join(os.path.dirname(current_dir), 'F1_data_mangement', filename)
    if os.path.exists(path2):
        try:
            return pd.read_csv(path2).replace(r'\\N', None, regex=True)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            return pd.DataFrame()
            
    return pd.DataFrame()

@app.route('/api/races/<int:year>', methods=['GET'])
def get_races_by_year(year):
    # Probeer eerst de grote dataset
    df = load_csv_data('unprocessed_f1_training_data.csv')
    
    if not df.empty:
        # Filter op jaar
        season_df = df[df['year'] == year]
        if not season_df.empty:
            # Unieke races halen
            races = season_df[['raceId', 'date', 'country']].drop_duplicates().sort_values('date')
            races['round'] = range(1, len(races) + 1)
            races['name'] = races['country'] # Gebruik land als naam
            return jsonify(races[['raceId', 'round', 'name', 'date']].to_dict(orient='records'))

    # Fallback naar races.csv als de grote dataset er niet is
    races_df = load_csv_data('races.csv')
    if not races_df.empty:
        season_races = races_df[races_df['year'] == year].copy()
        if not season_races.empty:
            season_races = season_races.sort_values('round')
            return jsonify(season_races[['raceId', 'round', 'name', 'date']].to_dict(orient='records'))

    return jsonify([])

@app.route('/api/wiki/<int:race_id>/<string:view_type>', methods=['GET'])
def get_wiki_data(race_id, view_type):
    df = load_csv_data('unprocessed_f1_training_data.csv')
    
    # Als unprocessed niet bestaat, probeer losse files (fallback)
    if df.empty:
        return jsonify([])

    # Filter op race
    race_df = df[df['raceId'] == race_id].copy()
    if race_df.empty: return jsonify([])

    data = []
    
    # Sorteren
    if view_type == 'race':
        race_df = race_df.sort_values('positionOrder')
    elif view_type == 'grid':
        race_df = race_df.sort_values('grid')
    elif view_type == 'qualifying':
        # Sorteer op quali positie als beschikbaar, anders grid
        if 'quali_position' in race_df.columns:
             race_df = race_df.sort_values('quali_position')
        else:
             race_df = race_df.sort_values('grid')

    for _, row in race_df.iterrows():
        # Driver naam netjes maken (bv. "hamilton" -> "Hamilton")
        driver_ref = str(row.get('driverRef', 'Unknown'))
        driver_name = driver_ref.capitalize()
        
        # Team naam
        team_name = row.get('name', 'Unknown') # In training data is 'name' vaak de constructor naam

        entry = {
            'driver': driver_name,
            'team': team_name,
        }

        if view_type == 'race':
            entry['position'] = row['positionOrder']
            entry['time'] = str(row['time']) if pd.notna(row['time']) else row.get('status', 'DNF')
            entry['points'] = row['points']
        
        elif view_type == 'grid':
            entry['position'] = row['grid']
            # Pak de beste tijd die beschikbaar is
            entry['time'] = str(row.get('q3', row.get('q2', row.get('q1', '-'))))

        elif view_type == 'qualifying':
            entry['position'] = row.get('quali_position', row.get('grid'))
            entry['q1'] = str(row.get('q1', '-'))
            entry['q2'] = str(row.get('q2', '-'))
            entry['q3'] = str(row.get('q3', '-'))

        data.append(entry)
        
    return jsonify(data)


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
        22: {"name": "Las Vegas", "circuit": "Las Vegas Strip"},
        23: {"name": "Qatar", "circuit": "Lusail"},
        24: {"name": "Abu Dhabi", "circuit": "Yas Island"},
    }
    return jsonify(races), 200


@app.route('/api/docs', methods=['GET'])
def get_docs_data():
    """Get data for the Documentation page"""
    return jsonify({
        'title': 'System Documentation',
        'sections': [
            {
                'id': 'models',
                'title': '🤖 AI Models',
                'content': 'This system uses a combination of XGBoost and Neural Networks to predict race outcomes.\n\n1. Pre-race Model: Analyzes historical data (2015-2024) including qualifying pace, driver form, and track history.\n2. Continuous Learner: Adapts to live lap times during the race to refine predictions.'
            },
            {
                'id': 'strategy',
                'title': '🛞 Tire Strategy',
                'content': 'Tire degradation is modeled per circuit using historical wear rates and current weather conditions.\n\nThe system calculates optimal pit windows based on:\n- Tire compound life (Soft/Medium/Hard)\n- Pit stop loss time (avg 20-24s)\n- Track temperature impact'
            },
            {
                'id': 'battle',
                'title': '⚔️ Battle Detector',
                'content': 'The Battle Detector monitors gaps between drivers in real-time.\n\n- Detection: When the gap is < 1.0s (DRS range).\n- Intensity: Calculated based on gap closure rate and position changes.\n- Events: Automatically generates alerts when overtakes are likely.'
            }
        ]
    }), 200


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
        
        # print(f"\n{'='*80}")
        # print(f"[PRERACE API] RACE {race_num}: {race_name} - Processing pre-race analysis")
        # print(f"{'='*80}")
        
        # Load model
        model = ensure_prerace_model_loaded()
        if not model or not model.loaded:
            print("[PRERACE API] ERROR: Model failed to load")
            return jsonify({'error': 'Could not load model'}), 500
        
        # Fetch grid using utility (P4.1 - caching opportunity identified)
        grid = get_race_grid(race_num)
        
        # Log grid positions for debugging
        # print(f"[PRERACE API] Grid positions for Race {race_num} ({race_name}):")
        # for i, driver in enumerate(grid[:10], 1):  # Show top 10
        #     print(f"    P{driver.get('grid_pos', i):2d}: {driver.get('driver'):3s} - {driver.get('team', 'Unknown')}")
        # if len(grid) > 10:
        #     print(f"    ... and {len(grid) - 10} more drivers")
        
        # print(f"[PRERACE API] Total: {len(grid)} drivers loaded")
        
        # Get predictions from model
        predictions = model.predict(grid, race_num)
        
        # print(f"[PRERACE API] ✓ Generated {len(predictions)} predictions for Race {race_num} ({race_name})")
        # print(f"[PRERACE API] Top 5 predictions:")
        # for i, pred in enumerate(predictions[:5], 1):
        #     print(f"    {i}. {pred.get('driver'):3s} (Grid P{pred.get('grid_position'):2d}) - Confidence: {pred.get('confidence', 0):.1f}%")
        # print(f"{'='*80}\n")
        
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
        
        # print(f"\n{'='*80}")
        # print(f"[TIRE STRATEGY API] RACE {race_num}: {race_name} - Computing tire strategy")
        # print(f"{'='*80}")
        
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
        
        # print(f"[TIRE STRATEGY API] Grid loaded: {len(grid)} drivers")
        
        # Get tire strategy predictions (includes per-driver strategies)
        strategies = tire_model.predict_strategy(grid, race_num, weather_forecast)
        
        # print(f"[TIRE STRATEGY API] ✓ Generated tire strategies for {len(strategies)} strategies")
        # print(f"[TIRE STRATEGY API] Top strategies:")
        # for i, strat in enumerate(strategies, 1):
        #     if strat.get('strategy_type') != 'per_driver_details':
        #         print(f"    {i}. {strat.get('strategy_type'):15s} - Confidence: {strat.get('confidence', 0):.0f}%")
        # print(f"{'='*80}\n")
        
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
        # print(f"  [GRID] Attempting to fetch FastF1 qualifying data for race {race_num}...")
        
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
            
            # print(f"  [GRID] ✓ Successfully loaded {len(grid)} drivers from FastF1 qualifying")
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
    
    # Removed outdated race_adjustments to prevent confusion.
    # If FastF1 fails, we use the base grid order (2024 standard)
    
    grid = []
    for i, driver in enumerate(base_grid):
        grid_pos = i + 1
        grid_pos = max(1, min(20, grid_pos))
        grid.append({
            'driver': driver['driver'],
            'number': driver['number'],
            'team': driver['team'],
            'grid_pos': grid_pos
        })
    
    # print(f"  [GRID] Using fallback grid for race {race_num}")
    # print(f"  [GRID] Top 10 drivers in fallback grid:")
    # sorted_grid = sorted(grid, key=lambda x: x['grid_pos'])[:10]
    # for driver in sorted_grid:
    #     print(f"    P{driver['grid_pos']:2d}: {driver['driver']:3s} - {driver['team']}")
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


@app.route('/api/race/replay-data', methods=['GET'])
def get_replay_data():
    """
    Get race replay data with track layout and frames for visualization
    Query params:
    - race: race number (1-22)
    
    Frames are cached to a JSON file to avoid regenerating on every request.
    First request generates and caches, subsequent requests load from cache.
    """
    try:
        race_num = int(request.args.get('race', 21))
        cache_dir = 'cache'
        cache_file = os.path.join(cache_dir, f'race_{race_num:02d}_frames.json')
        
        print(f"\n{'='*80}")
        print(f"[REPLAY API] Loading replay data for race {race_num}")
        print(f"{'='*80}")
        
        # Check if cached frames exist
        cached_data = None
        if os.path.exists(cache_file):
            try:
                print(f"[REPLAY API] Loading cached frames from {cache_file}...")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                print(f"[REPLAY API] [OK] Loaded {len(cached_data.get('frames', []))} cached frames")
                return jsonify(cached_data), 200
            except Exception as cache_err:
                print(f"[REPLAY API] WARNING: Could not load cache: {cache_err}")
                print(f"[REPLAY API] Regenerating frames...")
        
        # Fetch race data from FastF1
        fetcher = FastF1DataFetcher()
        if not fetcher.fetch_race(2024, race_num):
            raise Exception(f"Could not fetch race {race_num} from FastF1")
        
        # Get race info
        race_info = fetcher.get_race_summary()
        
        # Get track data for visualization
        print("[REPLAY API] Processing track data...")
        laps_data = fetcher.process_race_laps_streaming()
        
        if not laps_data:
            raise Exception("No lap data available for this race")
        
        # Build track geometry from telemetry (pass session for real coordinates)
        track_data = _build_track_geometry(laps_data, fetcher.session)
        
        # Get DRS zones from qualifying
        drs_zones = _get_drs_zones(race_num)
        
        # Build frames for animation (each lap, key positions)
        print(f"[REPLAY API] Building {len(laps_data)} frame data with 120fps ultra-smooth animation...")
        frames = _build_replay_frames(laps_data, race_info)
        
        print(f"[REPLAY API] ✓ Generated {len(frames)} frames")
        
        # Prepare response
        response_data = {
            'status': 'success',
            'raceName': race_info.get('event', f'Race {race_num}'),
            'year': 2024,
            'round': race_num,
            'trackData': track_data,
            'drsZones': drs_zones,
            'frames': frames,
            'totalFrames': len(frames),
        }
        
        # Cache frames to file for faster loading next time
        try:
            os.makedirs(cache_dir, exist_ok=True)
            print(f"[REPLAY API] Caching {len(frames)} frames to {cache_file}...")
            with open(cache_file, 'w') as f:
                json.dump(response_data, f, default=str)
            print(f"[REPLAY API] ✓ Frames cached successfully")
        except Exception as cache_write_err:
            print(f"[REPLAY API] WARNING: Could not cache frames: {cache_write_err}")
        
        print(f"{'='*80}\n")
        
        return jsonify(response_data), 200
        
    except Exception as e:
        print(f"[REPLAY API] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'error'}), 400


@app.route('/api/race/lap-predictions', methods=['GET'])
def get_lap_predictions():
    """
    Get real ML model win-probability predictions for every lap of a race.
    Runs ContinuousModelLearner incrementally (one lap at a time) so that
    predictions improve exactly as they do during a live simulation.

    Query params:
    - race: race number (1-22)

    Returns:
    {
      "status": "success",
      "race": 21,
      "total_laps": 58,
      "predictions_by_lap": {
        "1": [{"driver_code": "VER", "driver_name": "...", "team": "...",
                "confidence": 72.3, "predicted_position": 1}, ...],
        "2": [...],
        ...
      }
    }
    Results are cached to cache/race_XX_predictions.json after first computation.
    """
    try:
        race_num = int(request.args.get('race', 21))
        cache_dir_path = 'cache'
        cache_file = os.path.join(cache_dir_path, f'race_{race_num:02d}_predictions.json')

        # ── Serve from cache if available ──────────────────────────────────────
        if os.path.exists(cache_file):
            try:
                print(f"[PRED API] Loading cached predictions from {cache_file}")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return jsonify(json.load(f)), 200
            except Exception as cache_err:
                print(f"[PRED API] Cache read failed ({cache_err}), recomputing…")

        # ── Load session without telemetry (much faster) ──────────────────────
        # Do NOT use FastF1DataFetcher.fetch_race() here — that calls
        # session.load() which downloads full telemetry for all drivers.
        # For the ML model we only need lap/tire data, so skip telemetry entirely.
        print(f"[PRED API] Loading race {race_num} (laps only, no telemetry)...")
        race_session = fastf1.get_session(2024, race_num, 'R')
        race_session.load(laps=True, telemetry=False, weather=False, messages=False)

        # Build driver number -> 3-letter code map
        driver_code_map = {}
        try:
            for _, dr in race_session.results.iterrows():
                num = dr.get('DriverNumber')
                abbrev = str(dr.get('Abbreviation', ''))
                if num is not None and abbrev and abbrev != 'nan':
                    try:
                        driver_code_map[int(num)] = abbrev
                    except (TypeError, ValueError):
                        pass
        except Exception:
            pass

        # ── Build qualifying grid for grid_position feature ───────────────────
        qualifying_grid = {}
        try:
            qual = fastf1.get_session(2024, race_num, 'Q')
            qual.load(telemetry=False, weather=False)
            if qual.results is not None:
                for grid_pos, (_, row) in enumerate(qual.results.iterrows(), 1):
                    code = str(row.get('Abbreviation', ''))
                    if code and code != 'nan':
                        qualifying_grid[code] = grid_pos
        except Exception as qual_err:
            print(f"[PRED API] Qualifying grid load failed: {qual_err} — using fallback P15")

        # ── Fast lap extraction: no telemetry needed for the ML model ─────────
        session_laps = race_session.laps
        if session_laps is None or len(session_laps) == 0:
            raise Exception("No lap data in FastF1 session")

        laps_by_number = {}
        for _, row in session_laps.iterrows():
            lap_num_raw = row.get('LapNumber')
            position_raw = row.get('Position')

            try:
                lap_num = int(float(lap_num_raw))
                position = int(float(position_raw))
                driver_num_int = int(float(row.get('DriverNumber', 0)))
            except (TypeError, ValueError):
                continue
            if lap_num <= 0 or position <= 0 or driver_num_int <= 0:
                continue

            driver_num_str = str(driver_num_int)
            code = driver_code_map.get(driver_num_int, f'D{driver_num_str}')

            lap_time_raw = row.get('LapTime')
            if hasattr(lap_time_raw, 'total_seconds'):
                lap_time_s = float(lap_time_raw.total_seconds())
            else:
                try:
                    lap_time_s = float(lap_time_raw)
                except (TypeError, ValueError):
                    lap_time_s = 90.0
            if not (50 < lap_time_s < 300):
                lap_time_s = 90.0

            tire_age_raw = row.get('TyreLife')
            try:
                tire_age = int(float(tire_age_raw)) if tire_age_raw is not None else 1
            except (TypeError, ValueError):
                tire_age = 1

            compound = str(row.get('Compound', 'MEDIUM') or 'MEDIUM').upper()
            if compound in ('', 'NAN', 'UNKNOWN', 'NONE'):
                compound = 'MEDIUM'

            has_pit = bool(
                (row.get('PitInTime') is not None and str(row.get('PitInTime')) not in ('NaT', 'nan', 'None', '')) or
                (row.get('PitOutTime') is not None and str(row.get('PitOutTime')) not in ('NaT', 'nan', 'None', ''))
            )

            laps_by_number.setdefault(lap_num, []).append({
                'driver': code,
                'lap_number': lap_num,
                'position': position,
                'lap_time': lap_time_s,
                'tire_compound': compound,
                'tire_age': max(1, tire_age),
                'is_pit_lap': has_pit,
            })

        sorted_lap_nums = sorted(laps_by_number.keys())
        total_laps = max(sorted_lap_nums) if sorted_lap_nums else 58

        # ── Static driver metadata ────────────────────────────────────────────
        PRED_DRIVER_NAMES = {
            'VER': 'Max Verstappen',    'PER': 'Sergio Perez',
            'LEC': 'Charles Leclerc',   'SAI': 'Carlos Sainz',
            'HAM': 'Lewis Hamilton',    'RUS': 'George Russell',
            'NOR': 'Lando Norris',      'PIA': 'Oscar Piastri',
            'ALO': 'Fernando Alonso',   'STR': 'Lance Stroll',
            'GAS': 'Pierre Gasly',      'OCO': 'Esteban Ocon',
            'HUL': 'Nico Hulkenberg',   'MAG': 'Kevin Magnussen',
            'TSU': 'Yuki Tsunoda',      'RIC': 'Daniel Ricciardo',
            'ALB': 'Alexander Albon',   'SAR': 'Logan Sargeant',
            'BOT': 'Valtteri Bottas',   'ZHO': 'Zhou Guanyu',
        }
        PRED_DRIVER_TEAMS = {
            'VER': 'Red Bull Racing',   'PER': 'Red Bull Racing',
            'LEC': 'Ferrari',           'SAI': 'Ferrari',
            'HAM': 'Mercedes',          'RUS': 'Mercedes',
            'NOR': 'McLaren',           'PIA': 'McLaren',
            'ALO': 'Aston Martin',      'STR': 'Aston Martin',
            'GAS': 'Alpine',            'OCO': 'Alpine',
            'HUL': 'Haas F1 Team',      'MAG': 'Haas F1 Team',
            'TSU': 'RB',                'RIC': 'RB',
            'ALB': 'Williams',          'SAR': 'Williams',
            'BOT': 'Kick Sauber',       'ZHO': 'Kick Sauber',
        }

        # ── Run ContinuousModelLearner lap-by-lap ─────────────────────────────
        model = ContinuousModelLearner(total_race_laps=total_laps)
        predictions_by_lap = {}
        driver_pit_counts = {}   # track cumulative pit stops from is_pit_lap flag

        for lap_num in sorted_lap_nums:
            lap_records = laps_by_number[lap_num]

            # Build per-driver dicts for model input
            lap_drivers = []
            for rec in lap_records:
                code = rec.get('driver', 'UNK')
                # Accumulate pit stops whenever the lap has a pit event
                if rec.get('is_pit_lap', False):
                    driver_pit_counts[code] = driver_pit_counts.get(code, 0) + 1

                lap_time = rec.get('lap_time', 90.0)
                try:
                    lap_time = float(lap_time) if lap_time else 90.0
                except (TypeError, ValueError):
                    lap_time = 90.0

                tire_age = rec.get('tire_age', 1)
                try:
                    tire_age = int(float(tire_age)) if tire_age else 1
                except (TypeError, ValueError):
                    tire_age = 1

                lap_drivers.append({
                    'driver': code,
                    'position': int(rec.get('position', 20)),
                    'lap_time': lap_time,
                    'tire_compound': str(rec.get('tire_compound', 'MEDIUM')).upper(),
                    'tire_age': max(1, tire_age),
                    'pit_stops': driver_pit_counts.get(code, 0),
                    'grid_position': qualifying_grid.get(code, 15),
                    'points_constructor': 100.0,
                    'driver_age': 28,
                    'current_lap': lap_num,
                })

            if not lap_drivers:
                continue

            # Incremental model training
            model.add_race_data(lap_num, lap_drivers)
            model.update_model(lap_num)

            # Predict for this lap
            preds_dict = model.predict_lap(lap_drivers)

            lap_preds = []
            for code, pred_tuple in preds_dict.items():
                pred_pos, confidence = (
                    pred_tuple if isinstance(pred_tuple, tuple) else (15, 50.0)
                )
                lap_preds.append({
                    'driver_code': code,
                    'driver_name': PRED_DRIVER_NAMES.get(code, code),
                    'team': PRED_DRIVER_TEAMS.get(code, 'Unknown'),
                    'confidence': round(float(confidence), 1),
                    'predicted_position': int(round(float(pred_pos))),
                })

            lap_preds.sort(key=lambda x: x['confidence'], reverse=True)
            predictions_by_lap[str(lap_num)] = lap_preds

        result = {
            'status': 'success',
            'race': race_num,
            'total_laps': total_laps,
            'predictions_by_lap': predictions_by_lap,
        }

        # ── Cache to disk ─────────────────────────────────────────────────────
        try:
            os.makedirs(cache_dir_path, exist_ok=True)
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f)
            print(f"[PRED API] Predictions cached to {cache_file}")
        except Exception as cache_write_err:
            print(f"[PRED API] Cache write failed: {cache_write_err}")

        return jsonify(result), 200

    except Exception as e:
        print(f"[PRED API] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'error'}), 400


def _generate_procedural_track():
    """Generate a realistic F1 track circuit with straights and curves
    
    Creates a circuit similar to real F1 tracks with:
    - Main straights
    - Technical sections with multiple corners
    - High-speed sweepers
    """
    import math
    centerline = []
    
    # Generate a realistic circuit with multiple sections
    # Section 1: Main straight (bottom)
    for i in range(20):
        t = i / 20
        x = -400 + t * 800  # Straight from -400 to 400
        y = -250
        centerline.append({'x': x, 'y': y})
    
    # Section 2: High-speed curve (right)
    for i in range(15):
        t = i / 15
        angle = -math.pi / 2 + t * (math.pi / 3)
        x = 400 + 150 * math.cos(angle)
        y = -250 + 150 * math.sin(angle)
        centerline.append({'x': x, 'y': y})
    
    # Section 3: Technical section (top right)
    for i in range(10):
        t = i / 10
        x = 450 - t * 200
        y = -100 + t * 100
        centerline.append({'x': x, 'y': y})
    
    # Section 4: Top straight
    for i in range(20):
        t = i / 20
        x = 250 - t * 500
        y = 0
        centerline.append({'x': x, 'y': y})
    
    # Section 5: High-speed sweeper (top left)
    for i in range(15):
        t = i / 15
        angle = math.pi / 2 + t * (math.pi / 3)
        x = -400 + 150 * math.cos(angle)
        y = 0 + 150 * math.sin(angle)
        centerline.append({'x': x, 'y': y})
    
    # Section 6: Technical corner (left)
    for i in range(10):
        t = i / 10
        x = -450 + t * 200
        y = 150 - t * 100
        centerline.append({'x': x, 'y': y})
    
    # Section 7: Left straight (back to start)
    for i in range(10):
        t = i / 10
        x = -250
        y = 50 - t * 300
        centerline.append({'x': x, 'y': y})
    
    return centerline


def _extract_track_from_telemetry(session):
    """Extract real track geometry from FastF1 telemetry data
    
    Uses actual car positions from telemetry to build the true track layout.
    Telemetry is accessed via laps[i].telemetry which contains X, Y coordinates.
    """
    try:
        print("[TRACK] Extracting real telemetry coordinates from race laps...")
        
        # Get laps with telemetry
        if not hasattr(session, 'laps') or session.laps is None or len(session.laps) == 0:
            print("[TRACK] No laps available")
            return None
        
        all_laps = session.laps
        
        # Collect unique coordinates from first few laps (avoid redundant data)
        coords_set = set()
        coords_list = []
        
        # Get telemetry from first 10 laps (enough to map the circuit)
        sample_laps = min(10, len(all_laps))
        for lap_idx in range(sample_laps):
            # CRITICAL FIX: Use .loc[] to preserve Lap object (not .iloc[] which returns Series)
            try:
                lap_index = all_laps.index[lap_idx]
                actual_lap = all_laps.loc[lap_index]
                
                # Check if lap has telemetry method (Lap object)
                if not hasattr(actual_lap, 'get_telemetry'):
                    continue
                
                telemetry = actual_lap.get_telemetry()
                if telemetry is None or len(telemetry) == 0:
                    continue
                
                # Extract X, Y coordinates
                if 'X' in telemetry.columns and 'Y' in telemetry.columns:
                    # Sample every nth point to reduce data (avoid too many duplicates)
                    step = max(1, len(telemetry) // 100)  # ~100 points per lap
                    
                    for idx in range(0, len(telemetry), step):
                        row = telemetry.iloc[idx]
                        try:
                            x = float(row['X'])
                            y = float(row['Y'])
                            
                            # Only add if not seen before (avoid duplicates)
                            coord_tuple = (round(x, 1), round(y, 1))
                            if coord_tuple not in coords_set:
                                coords_set.add(coord_tuple)
                                coords_list.append({'x': x, 'y': y})
                        except (ValueError, TypeError):
                            continue
            except Exception as lap_err:
                continue
        
        if len(coords_list) > 50:  # Need minimum points for valid track
            print(f"[TRACK] ✓ Extracted {len(coords_list)} unique telemetry coordinates")
            return coords_list
        else:
            print(f"[TRACK] WARNING: Only got {len(coords_list)} coordinates, using fallback")
            return None
            
    except Exception as e:
        print(f"[TRACK] Error extracting telemetry: {e}")
        import traceback
        traceback.print_exc()
        return None


def _build_track_geometry(laps_data, session=None):
    """Extract track geometry from FastF1 telemetry or fallback to procedural
    
    Priority:
    1. Real telemetry coordinates from FastF1 session
    2. Fallback procedural track if telemetry unavailable
    """
    try:
        # First try to get real telemetry coordinates
        centerline = None
        if session is not None:
            centerline = _extract_track_from_telemetry(session)
        
        # Fallback to procedural track
        if not centerline or len(centerline) == 0:
            print("[TRACK] Using procedural track layout")
            centerline = _generate_procedural_track()
        
        if not centerline or len(centerline) == 0:
            return _get_fallback_track_data()
        
        # Calculate bounds
        xs = [p.get('x', 0) for p in centerline]
        ys = [p.get('y', 0) for p in centerline]
        
        bounds = {
            'minX': min(xs) if xs else -1000,
            'maxX': max(xs) if xs else 1000,
            'minY': min(ys) if ys else -1000,
            'maxY': max(ys) if ys else 1000,
        }
        
        return {
            'bounds': bounds,
            'centerline': centerline,
            'innerBoundary': _offset_points(centerline, -150),
            'outerBoundary': _offset_points(centerline, 150),
            'finishLine': _get_finish_line(centerline),
        }
    except Exception as e:
        print(f"[TRACK BUILD] Error building track: {e}")
        return _get_fallback_track_data()


def _offset_points(points, offset):
    """Offset track points perpendicular to centerline"""
    if len(points) < 2:
        return []
    
    result = []
    for i in range(len(points)):
        curr = points[i]
        next_p = points[(i + 1) % len(points)]
        
        # Calculate perpendicular direction
        dx = next_p['x'] - curr['x']
        dy = next_p['y'] - curr['y']
        length = (dx*dx + dy*dy) ** 0.5
        
        if length > 0:
            nx = -dy / length * offset
            ny = dx / length * offset
            result.append({'x': curr['x'] + nx, 'y': curr['y'] + ny})
        else:
            result.append(curr)
    
    return result


def _get_finish_line(centerline):
    """Get finish line coordinates from centerline"""
    if len(centerline) < 2:
        return None
    
    start = centerline[0]
    end = centerline[1]
    
    return {
        'start': start,
        'end': end,
    }


def _get_drs_zones(race_num):
    """Get DRS zones for the race"""
    try:
        # Attempt to get from qualifying session
        quali = fastf1.get_session(2024, race_num, 'Q')
        quali.load(telemetry=False, weather=False)
        
        if hasattr(quali, 'drs_zones') and quali.drs_zones:
            return [
                {
                    'name': f'DRS Zone {i+1}',
                    'points': [
                        {'x': float(z[0]), 'y': float(z[1])}
                        for z in zone
                    ],
                }
                for i, zone in enumerate(quali.drs_zones)
            ]
    except:
        pass
    
    return []


def _build_replay_frames(laps_data, race_info):
    """Build animation frames from lap telemetry with interpolation for smooth animation
    
    Creates multiple frames per lap with interpolated positions so drivers animate smoothly.
    CRITICAL: Uses qualifying grid for first frame to ensure correct starting positions.
    """
    frames = []
    frame_counter = 0
    frames_per_lap = 120  # 120 frames per lap = ultra-smooth 120fps animation (8280 total frames)
    
    # CRITICAL: Get actual qualifying grid to fix starting positions (not race lap 1!)
    print("[FRAMES] Loading qualifying grid for accurate starting positions...")
    qualifying_grid = {}
    try:
        qual = fastf1.get_session(2024, race_info.get('round', 21), 'Q')
        qual.load(telemetry=False, weather=False)
        
        if qual.results is not None:
            for grid_pos, (_, qual_row) in enumerate(qual.results.iterrows(), 1):
                driver_code = str(qual_row.get('Abbreviation', ''))
                if driver_code and driver_code != 'nan':
                    qualifying_grid[driver_code] = grid_pos
                    if grid_pos <= 5:
                        print(f"  P{grid_pos}: {driver_code}")
        
        if len(qualifying_grid) > 0:
            print(f"[FRAMES] ✓ Loaded qualifying grid with {len(qualifying_grid)} drivers")
        else:
            raise Exception("Qualifying results empty")
            
    except Exception as e:
        print(f"[FRAMES] WARNING: Could not load qualifying grid: {e}")
        print("[FRAMES] Will build grid from first lap race data...")
        # FALLBACK: Build qualifying grid from FIRST LAP positions
        if laps_data:
            first_lap_drivers = {}
            for lap in laps_data:
                lap_num = int(lap.get('lap_number', 0))
                if lap_num == 1:  # Only use first lap data
                    driver_code = str(lap.get('driver', ''))
                    if driver_code and driver_code not in first_lap_drivers:
                        first_lap_drivers[driver_code] = int(lap.get('position', 999))
            
            # Sort by position to get grid order
            sorted_by_pos = sorted(first_lap_drivers.items(), key=lambda x: x[1])
            for grid_pos, (code, _) in enumerate(sorted_by_pos, 1):
                qualifying_grid[code] = grid_pos
            
            print(f"[FRAMES] ✓ Built fallback grid from lap 1 with {len(qualifying_grid)} drivers")
    
    # Group laps by lap number
    DRIVER_NAMES = {
        'VER': 'Max Verstappen', 'LEC': 'Charles Leclerc', 'SAI': 'Carlos Sainz',
        'PIA': 'Oscar Piastri', 'NOR': 'Lando Norris', 'HAM': 'Lewis Hamilton',
        'RUS': 'George Russell', 'ALO': 'Fernando Alonso', 'STR': 'Lance Stroll',
        'GAS': 'Pierre Gasly', 'OCO': 'Esteban Ocon', 'MAG': 'Kevin Magnussen',
        'HUL': 'Nico Hulkenberg', 'BOT': 'Valtteri Bottas', 'ZHO': 'Zhou Guanyu',
        'TSU': 'Yuki Tsunoda', 'ALB': 'Alexander Albon', 'SAR': 'Logan Sargeant',
        'PER': 'Sergio Perez', 'RIC': 'Daniel Ricciardo',
    }
    
    # Group laps by lap number
    laps_by_number = {}
    for lap in laps_data:
        lap_num = int(lap.get('lap_number', 0))
        if lap_num not in laps_by_number:
            laps_by_number[lap_num] = []
        laps_by_number[lap_num].append(lap)
    
    sorted_laps = sorted(laps_by_number.keys())
    
    # Build complete list of ALL drivers in race (from all laps + qualifying grid)
    # This ensures every driver who raced is included, not just those in first lap
    all_drivers_in_race = set()
    
    # Add all drivers from qualifying grid (guaranteed to be complete)
    all_drivers_in_race.update(qualifying_grid.keys())
    
    # Add all drivers from lap data (catches anyone in actual race)
    for lap_records in laps_by_number.values():
        for record in lap_records:
            driver_code = str(record.get('driver', '')).strip()
            if driver_code and driver_code != 'UNK':
                all_drivers_in_race.add(driver_code)
    
    print(f"[DRIVERS] Total unique drivers in race: {len(all_drivers_in_race)}")
    print(f"[DRIVERS] From qualifying: {len(qualifying_grid)}, From race data: {len(all_drivers_in_race) - len(qualifying_grid)}")

    # ===== PRE-COMPUTE CUMULATIVE RACE TIMES =====
    # Sum each driver's lap times across ALL laps so we know their total elapsed
    # race time at every lap.  The gap from the leader at lap N is simply:
    #   cum_race_time[driver][N] - cum_race_time[leader][N]
    # This gives accurate, continuously-growing gaps (not just per-lap deltas).
    cumulative_race_times = {}   # code -> {lap_num: total_seconds_elapsed}
    for code in all_drivers_in_race:
        cumulative_race_times[code] = {}
        running = 0.0
        for ln in sorted_laps:
            lap_recs = laps_by_number.get(ln, [])
            found = next((r for r in lap_recs if r.get('driver') == code), None)
            if found:
                lt = found.get('lap_time')
                if isinstance(lt, (int, float)) and 50 < lt < 600:
                    running += float(lt)
            # Always store current running total so every lap has a value
            cumulative_race_times[code][ln] = running

    print(f"[FRAMES] Pre-computed cumulative race times for {len(cumulative_race_times)} drivers")

    # Generate frames with HIGH RESOLUTION interpolation for smooth 120fps playback
    # 120 frames per lap = 8280 total frames = true 120fps animation (cinema-smooth)
    frames_per_lap = 120  # 120 frames per lap = 120fps (ultra-smooth, 2x better than 60fps)
    frame_counter = 0
    frames = []
    
    # Track last known state for each driver (for retired drivers)
    last_driver_state = {}
    # Track last known X/Y for each driver so missing telemetry doesn't snap to (0,0)
    last_known_xy = {}
    
    # INITIALIZATION: Set initial positions from qualifying grid
    # This ensures all drivers start at their correct grid position in lap 1
    for code, grid_pos in qualifying_grid.items():
        last_driver_state[code] = {
            'driver': code,
            'position': grid_pos,  # Grid position = starting position
            'gear': 0,
            'throttle': 0,
            'brake': 0,
            'speed': 0,
            'drs_available': False,
            'tire_compound': 'MEDIUM',
            'tire_age': 0,
            'pit_stops': 0,
            'gap': '+0.000',
            'team': 'Unknown',
        }
    
    for lap_idx, lap_num in enumerate(sorted_laps):
        lap_records = laps_by_number[lap_num]
        
        # IMPORTANT: For ALL laps, use actual race lap data!
        # Even lap 1 should use race positions, not grid positions
        # Drivers start on grid at race start, but positions change immediately
        # We use qualifying_grid only as PREVIOUS positions for interpolation
        drivers_by_code = {}
        for record in lap_records:
            code = record.get('driver', 'UNK')
            # CRITICAL: Always use the LATEST record for this driver
            # (If multiple records exist, lap records are ordered, so last one is most current)
            drivers_by_code[code] = record
        
        # DEBUG: Write file to confirm we're processing laps
        if lap_num == 1:
            with open('lap_processing_debug.txt', 'w') as f:
                f.write(f"LAP {lap_num}: drivers_by_code has {len(drivers_by_code)} drivers\n")
                for code, record in list(drivers_by_code.items())[:3]:
                    f.write(f"  {code}: position={record.get('position')}, lap_time={record.get('lap_time')}, gap={record.get('gap')}\n")
            
        # Verify we have meaningful data
        if not drivers_by_code:
            continue  # Skip lap with no driver data
        
        # Get previous lap drivers for interpolation
        prev_drivers = {}
        if lap_idx > 0:
            prev_lap_num = sorted_laps[lap_idx - 1]
            prev_lap_records = laps_by_number[prev_lap_num]
            for record in prev_lap_records:
                code = record.get('driver', 'UNK')
                if code not in prev_drivers:
                    prev_drivers[code] = record
        else:
            # LAP 1: Use qualifying grid as "previous positions" for smooth interpolation from grid to race positions
            # This ensures drivers animate from grid position P1, P2, P3 to their actual lap 1 race positions
            for code, grid_pos in qualifying_grid.items():
                prev_drivers[code] = {
                    'driver': code,
                    'position': grid_pos,  # Use grid position as the previous position for interpolation
                    'gear': 0,
                    'throttle': 0,
                    'brake': 0,
                    'speed': 0,
                }
            for code, grid_pos in qualifying_grid.items():
                prev_drivers[code] = {
                    'driver': code,
                    'position': grid_pos,  # Use grid position as the previous position for interpolation
                    'gear': 0,
                    'throttle': 0,
                    'brake': 0,
                    'speed': 0,
                }
        
        # ALWAYS generate full frames per lap for smooth continuous animation
        frames_to_generate = frames_per_lap

        # --- Pre-compute leader lap time and per-driver gap fractions --------
        # Use pre-computed CUMULATIVE race times so every driver's gap grows
        # correctly over the whole race (not just this lap's time delta, and
        # never stale '+0.000' values from FastF1 which has no Gap column).
        leader_lap_time_s = None
        leader_cum_time = 0.0
        leader_code_this_lap = None
        driver_gap_fraction = {}   # code -> float (0..1), how far behind leader

        # Find the leader (P1) and their lap time + cumulative race time
        for code, record in drivers_by_code.items():
            pos = int(record.get('position', 999) or 999)
            if pos == 1:
                leader_code_this_lap = code
                lt = record.get('lap_time')
                if isinstance(lt, (int, float)) and 50 < lt < 600:
                    leader_lap_time_s = float(lt)
                leader_cum_time = cumulative_race_times.get(code, {}).get(lap_num, 0.0)
                break

        if leader_lap_time_s is None:
            leader_lap_time_s = 90.0   # safe fallback (1:30 lap)

        # Compute gap fraction for every driver using cumulative race times.
        # Use lap_num - 1 (PREVIOUS lap) so that at the START of lap N, drivers
        # are spread out by the gap they built up through lap N-1.
        # For lap 1, prev_lap = 0 → no data → gap = 0 → everyone at start line.
        prev_lap_for_gap = lap_num - 1
        leader_cum_prev = cumulative_race_times.get(leader_code_this_lap, {}).get(prev_lap_for_gap, 0.0)
        for code in all_drivers_in_race:
            driver_cum = cumulative_race_times.get(code, {}).get(prev_lap_for_gap, None)
            if driver_cum is not None and leader_cum_prev >= 0:
                gap_s = max(0.0, driver_cum - leader_cum_prev)
            else:
                gap_s = 0.0
            # Fraction of one lap that this driver is behind the leader.
            # Modulo handles lapped cars: they wrap back around on the track.
            frac = (gap_s % leader_lap_time_s) / leader_lap_time_s
            driver_gap_fraction[code] = frac

        # Create frames
        for frame_step in range(frames_to_generate):
            # Leader's progress through the lap: 0.0 → 1.0
            t_leader = frame_step / frames_to_generate if frames_to_generate > 0 else 0

            drivers = {}

            # IMPORTANT: Iterate through ALL drivers in race, not just current lap drivers
            for code in all_drivers_in_race:
                current_record = drivers_by_code.get(code)
                prev_record = prev_drivers.get(code)

                if not current_record and not prev_record and code not in last_driver_state:
                    continue

                if not current_record:
                    if code in last_driver_state:
                        current_record = last_driver_state[code]
                    elif prev_record:
                        current_record = prev_record
                    else:
                        continue

                current_pos = int(current_record.get('position', 20) or 20)
                current_gear = int(current_record.get('gear') or 0)
                current_throttle = float(current_record.get('throttle') or 0)
                current_brake = float(current_record.get('brake') or 0)
                current_speed = float(current_record.get('speed') or 0)

                # ===== TELEMETRY POINT SELECTION =====
                # Adjust t by this driver's gap fraction so they appear behind
                # the leader by the correct track distance.
                gap_frac = driver_gap_fraction.get(code, 0.0)
                t_driver = t_leader - gap_frac
                # Wrap negative values: driver is still on the track,
                # just earlier in the same lap cycle.
                if t_driver < 0:
                    t_driver += 1.0
                t_driver = max(0.0, min(1.0, t_driver))

                telemetry_points = current_record.get('telemetry_points', [])
                current_x = None
                current_y = None
                current_tel_speed = None

                # ── Sanitize telemetry: remove consecutive duplicate positions ──
                # FastF1 sometimes produces "frozen" runs of identical (x,y) with
                # speed=0 due to sensor gaps or sparse data. Keeping only one copy
                # of each unique position prevents drivers from appearing frozen on
                # track for long stretches before teleporting to the next point.
                if len(telemetry_points) > 2:
                    sanitized = [telemetry_points[0]]
                    for pt in telemetry_points[1:]:
                        prev_pt = sanitized[-1]
                        if pt.get('x') != prev_pt.get('x') or pt.get('y') != prev_pt.get('y'):
                            sanitized.append(pt)
                    if len(sanitized) >= 2:
                        telemetry_points = sanitized

                if len(telemetry_points) > 0:
                    tel_idx = int(t_driver * (len(telemetry_points) - 1))
                    tel_idx = max(0, min(len(telemetry_points) - 1, tel_idx))

                    point = telemetry_points[tel_idx]
                    current_x = point.get('x')
                    current_y = point.get('y')
                    current_tel_speed = point.get('speed')
                    if current_tel_speed is None:
                        current_tel_speed = current_speed

                # Position interpolation (leaderboard only, not track x,y)
                prev_pos = None
                if prev_record:
                    try:
                        prev_pos = int(prev_record.get('position') or current_pos)
                    except (TypeError, ValueError):
                        prev_pos = current_pos
                if prev_pos is None:
                    prev_pos = int(last_driver_state[code].get('position', current_pos)) \
                               if code in last_driver_state else current_pos

                if t_leader < 0.3:
                    eased_t = (t_leader / 0.3) ** 2 * 0.3
                else:
                    remaining_t = (t_leader - 0.3) / 0.7
                    eased_t = 0.3 + (1 - (1 - remaining_t) ** 2) * 0.7

                position = prev_pos + (current_pos - prev_pos) * eased_t
                position = max(1.0, min(20.0, position))

                # ===== TELEMETRY X,Y =====
                xy_valid = (current_x is not None and current_y is not None
                            and not (current_x == 0 and current_y == 0))
                if xy_valid:
                    x = current_x
                    y = current_y
                    last_known_xy[code] = (x, y)
                elif code in last_known_xy and last_known_xy[code] != (0, 0):
                    x, y = last_known_xy[code]
                else:
                    x, y = None, None

                # ===== TELEMETRY DATA =====
                if len(telemetry_points) > 0:
                    point = telemetry_points[tel_idx]
                    tel_speed    = point.get('speed',    current_tel_speed)
                    tel_gear     = point.get('gear',     current_gear)
                    tel_throttle = point.get('throttle', current_throttle)
                    tel_brake    = point.get('brake',    current_brake)
                else:
                    tel_speed    = current_tel_speed
                    tel_gear     = current_gear
                    tel_throttle = current_throttle
                    tel_brake    = current_brake
                
                # Skip drivers with no position data at all
                if x is None or y is None:
                    continue
                
                drivers[code] = {
                    'code': code,
                    'driver_name': DRIVER_NAMES.get(code, code),
                    'position': position,
                    'x': x,
                    'y': y,
                    'speed': tel_speed if tel_speed is not None else current_speed,
                    'gear': tel_gear if tel_gear is not None else current_gear,
                    'throttle': tel_throttle if tel_throttle is not None else current_throttle,
                    'brake': tel_brake if tel_brake is not None else current_brake,
                    'drs': bool(current_record.get('drs_available', False)),
                    'tire_compound': str(current_record.get('tire_compound', 'MEDIUM')),
                    'tire_age': int(current_record.get('tire_age', 0) or 0),
                    'pit_stops': int(current_record.get('pit_stops', 0) or 0),
                    'gap': current_record.get('gap', '+0.000'),
                    'team': current_record.get('team', 'Unknown'),
                    'status': 'Running',
                }
                
                # CRITICAL: Cache this driver's state for retired drivers
                # This ensures retired drivers stay visible at their last position
                last_driver_state[code] = current_record
            
            # ===== CALCULATE GAPS USING CUMULATIVE RACE TIMES =====
            # Use the same cumulative_race_times we pre-computed before the lap
            # loop.  This gives the TRUE gap from race start, not just the
            # single-lap time delta.  Precision kept at 3 decimal places.
            leader_cum_display = cumulative_race_times.get(
                leader_code_this_lap, {}).get(lap_num, None)

            for code, driver_data in drivers.items():
                if leader_code_this_lap and code == leader_code_this_lap:
                    driver_data['gap'] = '+0.000'
                elif leader_cum_display is not None:
                    driver_cum = cumulative_race_times.get(code, {}).get(lap_num, None)
                    if driver_cum is not None:
                        gap_s = max(0.0, driver_cum - leader_cum_display)
                        driver_data['gap'] = f"+{gap_s:.3f}"
                    else:
                        driver_data['gap'] = None
                else:
                    driver_data['gap'] = None
            
            frames.append({
                'frameIndex': frame_counter,
                'lap': lap_num,
                'raceTime': (lap_num - 1) * 90 + (frame_step / frames_per_lap) * 90,
                'trackStatus': 'GREEN',
                'drivers': drivers,
                'predictions': [],
            })
            
            frame_counter += 1
    
    print(f"[FRAMES] Generated {len(frames)} smooth 120fps frames from {len(sorted_laps)} laps ({frames_per_lap} frames per lap)")
    return frames
    return frames


def _get_fallback_track_data():
    """Fallback track data if real data unavailable"""
    return {
        'bounds': {
            'minX': -1000,
            'maxX': 1000,
            'minY': -1000,
            'maxY': 1000,
        },
        'centerline': [],
        'innerBoundary': [],
        'outerBoundary': [],
        'finishLine': None,
    }


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
            # STEP 1: Get drivers and grid positions from RACE session (GridPosition column)
            # We use the actual GridPosition from the race data (includes penalties)
            print("[BACKGROUND] Loading drivers and grid positions from Race session...")
            drivers = []
            session = fetcher.session  # Get session reference
            laps = None
            
            try:
                if hasattr(session, 'results') and session.results is not None and len(session.results) > 0:
                    # Use GridPosition from Race results (accounts for penalties)
                    for _, row in session.results.iterrows():
                        driver_code = str(row.get('Abbreviation', ''))
                        if driver_code and driver_code != 'nan':
                            # GridPosition is float, 0.0 usually means pit lane or issue
                            grid_pos = float(row.get('GridPosition', 0.0))
                            
                            # Handle pit lane starts (0.0) -> put them at the back
                            if grid_pos <= 0.1:
                                grid_pos = 20.0 # Default to back
                            
                            driver_name = str(row.get('FullName', 'Unknown'))
                            team_name = str(row.get('TeamName', 'Unknown'))
                            driver_num = int(row.get('DriverNumber', 0))
                            
                            drivers.append({
                                'code': driver_code,
                                'name': driver_name,
                                'team': team_name,
                                'number': driver_num,
                                'grid_position': int(grid_pos)
                            })
                    
                    # Sort by grid position
                    drivers.sort(key=lambda x: x['grid_position'])
                    
                    # Print for debugging
                    for d in drivers:
                        print(f"    Grid P{d['grid_position']:2d}: {d['code']} - {d['name']}")
                    
                    print(f"[BACKGROUND] OK: Loaded {len(drivers)} drivers from RACE results")
                else:
                     print("[BACKGROUND] WARNING: No race results found")
            except Exception as e:
                print(f"[BACKGROUND] ERROR loading drivers from race: {e}")
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
    print(f"[SOCKETIO] INFO: CLIENT DISCONNECTED: {request.sid}")


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
                    'weather': lap_state.get('weather', {}),
                    'track_status': lap_state.get('track_status', '1'),
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
                    'weather': lap_state.get('weather', {}),
                    'track_status': lap_state.get('track_status', '1'),
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