# F1 AI Tracker - Copilot Instructions

## Project Overview

F1 AI Tracker is a **real-time F1 race prediction system** with per-lap continuous learning. It combines:
- **Backend**: Flask + SocketIO server with incremental ML model training
- **Frontend**: React + Vite SPA with WebSocket real-time updates
- **Data**: FastF1 API integration + historical training data (5-year baseline)

## Architecture & Data Flow

### Core Components

1. **Data Layer** (`fastf1_data_fetcher.py`)
   - Fetches live F1 race sessions from FastF1 API
   - Caches data locally in system temp directory
   - Extracts lap-by-lap telemetry for all drivers
   - Returns structured DataFrames with driver positions, times, tire info

2. **ML Core** (`continuous_model_learner.py`)
   - **Incremental learning**: Uses `SGDRegressor` with `partial_fit()` per lap
   - Pre-trained on historical 5-year dataset (`f1_historical_5years.csv`)
   - Ensemble approach: SGD + MLP + GradientBoosting predictions
   - **Critical design**: Confidence scores capped at 85% (no deterministic predictions)
   - Features: grid position, driver age, constructor points, circuit ID, tire data

3. **Race Simulation** (`race_simulator.py`)
   - Lap-by-lap simulation runner with adjustable speed
   - Manages driver states (position, tire age, pit stops)
   - Orchestrates model predictions every lap
   - Emits real-time updates via SocketIO

4. **Flask Backend** (`app.py`)
   - REST API: `/api/health`, `/api/races`, `/api/race/init`
   - SocketIO events for live race control: `race/start`, `race/pause`, `race/speed`
   - Manages global `race_state` dict (shared between threads)
   - Async threading model for WebSocket updates

5. **React Frontend** (`frontend/src/`)
   - Uses `socket.io-client` to listen for lap updates
   - Components: `RaceSelector`, `PredictionsPanel`, `DriversList`, `RaceInfo`
   - Vite dev server on default port (check `frontend/vite.config.js`)

## Key Developer Workflows

### Starting the Project

```bash
# 1. Start backend (Python environment required)
python app.py
# Runs on http://localhost:5000

# 2. In another terminal, start frontend
cd frontend && npm run dev
# Runs on http://localhost:5173 (or next available port)
```

### Testing a Single Component

- **Backend ML**: `python test_improved_model.py`
- **Data fetching**: `python fastf1_data_fetcher.py`
- **Race prediction**: `python race_predictor.py` (standalone CLI)

### Output Files

- Race predictions saved to `outputs/race_XX_YYYYMMDD_HHMMSS.txt`
- Historical data: `f1_historical_5years.csv`, `processed_f1_training_data.csv`

## Project-Specific Conventions

### Model Training Philosophy

- **Never deterministic**: Predictions use RandomForestClassifier for probabilistic top-5 rankings
- **Confidence formula**: Base 72% + pace spread (+10% max) + model maturity (+3%) - volatility penalty (−3% to −9%) = **capped at 85%**
- **Per-lap updates**: Model trains on EVERY lap (not batch processing), enabling real-time improvement

### SocketIO Communication Pattern

All driver objects in lap updates follow this structure:
```javascript
{
  position: 1,
  driver_code: "VER",
  driver_name: "Max Verstappen",
  team: "Red Bull Racing",
  lap_time: "1:28.473",
  tire_compound: "SOFT|MEDIUM|HARD",
  tire_age: 5,        // laps on current tire
  pit_stops: 1,
  gap: "+0.000",      // gap to P1
  laps_completed: 58
}
```

### Feature Engineering

Model uses these features (see `continuous_model_learner.py`):
- Grid position (starting position)
- Driver age
- Constructor points (team strength)
- Circuit ID (track-specific performance)
- Tire compound + age
- Historical performance data

### Caching Strategy

- FastF1 data cached in `{TEMP_DIR}/fastf1_cache/`
- Cross-platform: Uses `tempfile.gettempdir()` for compatibility
- Fallback training: Uses `processed_f1_training_data.csv` if main historical file missing

## Critical Integration Points

### Backend ↔ Frontend Communication

**Client → Server Events**:
- `race/start` with `{speed: 1.0}`
- `race/pause`, `race/resume`
- `race/speed` with `{speed: 2.0}`

**Server → Client Events**:
- `lap/update`: `{lap_number, drivers[], predictions[], events[]}`
- `race/finished`: `{final_standings}`
- `connect_response`: Initial connection confirmation

### Global State Management

Backend uses shared `race_state` dict (managed via threading locks):
- `running`: boolean
- `current_lap`: integer
- `predictions`: list of top-5 drivers with confidence scores
- `simulation_speed`: float (1.0 = real-time)

## Common Pitfalls & Solutions

| Issue | Root Cause | Solution |
|-------|-----------|----------|
| Model 100% confidence | Regression not capped | Check `ContinuousModelLearner.get_confidence_scores()` caps at 85% |
| WebSocket connection fails | CORS/async_mode mismatch | Backend uses `async_mode='threading'` and `CORS(origins="*")` |
| FastF1 API timeout | No cache enabled | Verify cache enabled in `fastf1_data_fetcher.py` line ~19 |
| Missing features in training | Fallback CSV incomplete | Check both `f1_historical_5years.csv` and `processed_f1_training_data.csv` exist |

## External Dependencies

- **FastF1**: F1 data API (requires internet for initial fetch, then cached)
- **XGBoost, scikit-learn**: ML models
- **Flask/SocketIO**: Real-time backend
- **React 19 + Vite**: Frontend build
- **socket.io-client**: Frontend WebSocket client

## Reference Entry Points

- Start here: [README.md](../README.md)
- Architecture deep-dive: [BACKEND_README.md](../BACKEND_README.md)
- Model details: [MODEL_IMPROVEMENTS.md](../MODEL_IMPROVEMENTS.md)
- Main CLI: [race_predictor.py](../race_predictor.py)
- Backend main: [app.py](../app.py)
