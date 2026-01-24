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
from continuous_model_learner import ContinuousModelLearner
from fastf1_data_fetcher import FastF1DataFetcher
import pandas as pd
import fastf1
from race_simulator import RaceSimulator

# Setup Flask App
app = Flask(__name__)

# Configureer CORS
CORS(app, supports_credentials=True, origins="*")

# Socket.IO setup - use polling only (avoid werkzeug WebSocket issues)
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
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

# Model cache
model_cache = {
    'model': None,
    'loaded': False
}


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


@app.route('/api/race/init', methods=['POST', 'GET'])
def init_race():
    """Initialize a race simulation"""
    try:
        # Support both POST (with body) and GET (with query param)
        if request.method == 'POST':
            race_num = request.json.get('race_number', 21)
        else:
            race_num = int(request.args.get('race', 21))
        
        print(f"[BACKEND] Initializing race {race_num}...")
        
        # Fetch REAL race data from FastF1
        drivers = None
        laps = None
        weather_data = None
        
        # Try to fetch real FastF1 data
        try:
            fetcher = FastF1DataFetcher()
            # Use 2024 season as default
            if fetcher.fetch_race(2024, race_num):
                # Get qualifying session for grid positions
                try:
                    qual_session = fastf1.get_session(2024, race_num, 'Q')
                    qual_session.load()
                    qual_results = qual_session.results
                except:
                    qual_results = None
                
                # Get real drivers from FastF1 race session
                driver_codes = fetcher.get_drivers_in_race()
                
                # Get session data for driver info
                session = fetcher.session
                drivers = []
                
                # Build driver list from FastF1 session results
                if hasattr(session, 'results') and session.results is not None:
                    results = session.results
                    for idx, (code, row) in enumerate(results.iterrows()):
                        if pd.notna(row.get('Abbreviation')):
                            driver_code = str(row.get('Abbreviation', code))
                            
                            # Get grid position from qualifying if available
                            grid_pos = idx + 1  # Default fallback
                            if qual_results is not None:
                                try:
                                    # Find driver in qualifying results
                                    qual_row = qual_results[qual_results['Abbreviation'] == driver_code]
                                    if len(qual_row) > 0:
                                        grid_pos = int(qual_row.iloc[0]['GridPosition'])
                                except:
                                    pass
                            
                            drivers.append({
                                'code': driver_code,
                                'name': str(row.get('FullName', 'Unknown')),
                                'team': str(row.get('TeamName', 'Unknown')),
                                'number': int(row.get('DriverNumber', idx + 1)),
                                'grid_position': grid_pos
                            })
                
                # Get real lap data
                if hasattr(session, 'laps') and session.laps is not None:
                    laps = session.laps
                
                # Get weather data from session
                weather_data = None
                try:
                    if hasattr(session, 'weather_data') and session.weather_data is not None:
                        weather_data = session.weather_data
                        if len(weather_data) > 0:
                            latest_weather = weather_data.iloc[-1] if isinstance(weather_data, pd.DataFrame) else weather_data[-1]
                            print(f"[BACKEND] ✅ Weather data loaded")
                except:
                    pass
                
                if len(drivers) > 0:
                    print(f"[BACKEND] ✅ Loaded {len(drivers)} drivers from FastF1")
                else:
                    raise Exception("No drivers found in FastF1 data")
                    
            else:
                raise Exception(f"Failed to fetch race {race_num} from FastF1")
        except Exception as e:
            print(f"[BACKEND] ⚠️ Could not load FastF1 data: {e}")
            print(f"[BACKEND] Falling back to dummy drivers...")
            
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
        
        print(f"[BACKEND] ✅ Loaded {len(drivers)} drivers")
        
        # Load and train AI model with real data
        try:
            print("[BACKEND] Loading AI model...")
            model = ContinuousModelLearner()
            model_cache['model'] = model
            
            # Pre-train on historical data if available
            if laps is not None and len(laps) > 0:
                print(f"[BACKEND] Pre-training AI model on {len(laps)} laps...")
                # Train model on real lap data
                try:
                    model.update_model(laps)  # Use update_model, not update_with_lap_data
                except Exception as train_err:
                    print(f"[BACKEND] ⚠️ Could not train model: {train_err}")
            
            model_cache['loaded'] = True
            print("[BACKEND] ✅ AI model ready!")
        except Exception as e:
            print(f"[BACKEND] Error loading AI model: {str(e)}")
            model_cache['model'] = None
            model_cache['loaded'] = True  # Still allow race to start
        
        # Initialize race simulator - wrapped in try/catch
        try:
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
        except Exception as sim_err:
            # Fallback if RaceSimulator fails
            print(f"[BACKEND] RaceSimulator failed: {sim_err}, using simple state")
            race_state['drivers'] = drivers
            race_state['race_name'] = f'Race {race_num}'
            race_state['total_laps'] = 58
            race_state['current_lap'] = 0
            race_state['race_simulator'] = None  # Mark as failed but continue
        
        print(f"[BACKEND] Race initialized! {len(race_state['drivers'])} drivers")
        
        return jsonify({
            'status': 'initialized',
            'race_id': 1,  # Simple race ID
            'race_name': race_state['race_name'],
            'total_laps': race_state['total_laps'],
            'drivers': race_state['drivers']
        }), 200
        
    except Exception as e:
        error_msg = str(e)
        print(f"[BACKEND] ❌ ERROR initializing race: {error_msg}")
        import traceback
        tb = traceback.format_exc()
        print(tb)
        return jsonify({'error': error_msg, 'traceback': tb}), 500


# HTTP endpoints for race control (instead of Socket.IO)
@app.route('/api/race/state', methods=['GET'])
def get_race_state():
    """Get current race state"""
    return jsonify({
        'lap_number': race_state['current_lap'],
        'current_lap': race_state['current_lap'],
        'total_laps': race_state['total_laps'],
        'drivers': race_state['drivers'],
        'predictions': race_state['predictions'],
        'running': race_state['running'],
        'weather': race_state.get('weather', {})
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
    print(f"[SOCKETIO] ✅ CLIENT CONNECTED")
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
    print(f"[SOCKETIO] ❌ CLIENT DISCONNECTED: {request.sid}")


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
    """Main simulation loop - runs in background thread"""
    while race_state['running'] and race_state['current_lap'] <= race_state['total_laps']:
        try:
            # Get next lap state from simulator
            lap_state = race_state['race_simulator'].simulate_lap(race_state['current_lap'])
            
            # UPDATE race_state with fresh data so HTTP polling clients see changes
            race_state['drivers'] = lap_state['drivers']
            race_state['predictions'] = lap_state['predictions']
            race_state['weather'] = lap_state.get('weather', race_state.get('weather', {}))
            
            print(f"[SIMULATION] Lap {race_state['current_lap']} updated - {len(race_state['drivers'])} drivers")
            
            # Emit lap update to any Socket.IO clients
            socketio.emit('lap/update', {
                'lap_number': race_state['current_lap'],
                'drivers': lap_state['drivers'],
                'predictions': lap_state['predictions'],
                'events': lap_state.get('events', []),
                'weather': lap_state.get('weather', {})
            }, to=None)
            
            # Move to next lap
            race_state['current_lap'] += 1
            
            # Simulate delay based on speed
            # Speed 1.0 = 5 seconds per lap (fast initial demo)
            # Speed 2.0 = 2.5 seconds per lap
            # Speed 0.5 = 10 seconds per lap
            delay = 5.0 / race_state['simulation_speed']
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
    
    # Run with werkzeug
    socketio.run(
        app, 
        host='127.0.0.1', 
        port=5000, 
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=False
    )
