# F1 AI Tracker

**Final Work 2026 - Erasmus Hogeschool Brussel**

---

## Project Goal

The goal of this project is to build a real-time Formula 1 race prediction system that combines live race data with machine learning. The system replays real F1 races lap by lap, trains an AI model continuously on incoming data, and displays live predictions, battle detection, and race events through an interactive web dashboard.

This is not a static analysis tool. The model learns during the race, improving its predictions with every lap, exactly like a real-time decision support system would work.

---

## What We Built

The project consists of a Python backend and a React frontend connected via WebSockets.

**Backend (Python / Flask)**
- Fetches real lap-by-lap race data from the FastF1 API for all 22 races of the 2024 F1 season
- Trains a machine learning model (SGDRegressor + ensemble) incrementally on each new lap using `partial_fit()`
- Pre-trains the model on 5 years of historical F1 data before the race starts
- Simulates a race replay at adjustable speed (1x, 2x, 5x)
- Detects battles between drivers in real time based on gap thresholds
- Detects overtakes, failed overtake attempts, and pit stops
- Calculates overtake probability per driver pair based on tire age, driver aggression, and circuit overtaking difficulty
- Emits live updates to the frontend via Socket.IO after every lap

**Frontend (React / Vite)**
- Live race dashboard with positions, lap times, tire compounds, gap data, and pit stops
- Predictions panel showing the top 5 predicted race finishers with confidence scores
- Notifications panel showing real-time race events (battles, overtakes, pit stops)
- Battle widget showing active on-track fights with overtake probability
- Position history chart tracking driver movements over the race
- Pre-race analysis page with predicted podium, tire strategy, and driver statistics
- Race replay page for reviewing completed simulations
- Weather widget, model metrics panel, and track renderer
- Dark/light theme support

**Machine Learning**
- Pre-race model: XGBoost trained on historical 2015-2024 data to predict the starting grid outcome
- Live model: `ContinuousModelLearner` using `SGDRegressor` with `partial_fit()` for incremental lap-by-lap learning
- Features used: grid position, driver age, constructor points, circuit ID, tire compound, tire age, pit stop count
- Confidence scores are capped at 85% to prevent deterministic overconfidence

---

## Project Structure

```
F1-AI-tracker/
|
|-- app.py                          # Flask + SocketIO backend server
|-- race_simulator.py               # Lap-by-lap race simulation engine
|-- continuous_model_learner_v2.py  # Incremental ML model (live learning)
|-- fastf1_data_fetcher.py          # FastF1 API integration and data fetching
|-- battle_detector.py              # Real-time battle and overtake detection
|-- event_generator.py              # Race event formatting for notifications
|-- race_predictor.py               # Standalone CLI race predictor
|
|-- model/
|   |-- prerace_model.py            # XGBoost pre-race prediction model
|   |-- tire_strategy_model.py      # Tire strategy ML model
|   |-- tire_strategy_ml.py         # Tire strategy training pipeline
|
|-- data/
|   |-- f1_historical_5years.csv    # Historical race data 2019-2024
|   |-- processed_f1_training_data.csv
|   |-- f1_weather_data.csv
|
|-- models/
|   |-- compound_model.json         # Trained tire compound model
|   |-- pit_stop_model.json         # Trained pit stop model
|   |-- stops_model.json
|
|-- frontend/
|   |-- src/
|   |   |-- pages/                  # Dashboard, PreRaceAnalysis, RaceReplay, Wiki, Docs
|   |   |-- components/             # All UI components
|   |   |-- services/               # API and WebSocket client
|   |   |-- utils/                  # Team colors, helpers
|   |-- vite.config.js
|   |-- package.json
|
|-- analysis/                       # Exploratory data analysis scripts
|-- scripts/                        # Data preparation scripts
|-- cache/                          # FastF1 cache and race frame data
|-- requirements.txt
```

---

## How It Works

1. The user selects a race from the 2024 F1 season in the dashboard
2. The backend fetches all lap data for that race from the FastF1 API (cached after first fetch)
3. The pre-race model generates an initial prediction based on qualifying and historical data
4. The simulation starts and processes one lap at a time at the selected speed
5. After each lap, the live model trains on the new data using `partial_fit()`
6. Battle detection runs on the top 5 drivers, checking gaps and position changes
7. All updates (positions, predictions, events) are emitted to the frontend via Socket.IO
8. The frontend renders the update in real time

---

## Getting Started

**Requirements**

- Python 3.10 or higher
- Node.js 18 or higher

**Backend**

```bash
pip install -r requirements.txt
python app.py
```

The backend runs on `http://localhost:5000`.

**Frontend**

```bash
cd frontend
npm install
npm run dev
```

The frontend runs on `http://localhost:5173`.

---

## Available Races (2024 Season)

| Nr. | Race              | Nr. | Race            |
|-----|-------------------|-----|-----------------|
| 1   | Bahrain           | 12  | United Kingdom  |
| 2   | Saudi Arabia      | 13  | Hungary         |
| 3   | Australia         | 14  | Belgium         |
| 4   | Japan             | 15  | Netherlands     |
| 5   | China             | 16  | Italy           |
| 6   | Miami             | 17  | Azerbaijan      |
| 7   | Imola             | 18  | Singapore       |
| 8   | Monaco            | 19  | Austin          |
| 9   | Canada            | 20  | Mexico          |
| 10  | Spain             | 21  | Brazil          |
| 11  | Austria           | 22  | Abu Dhabi       |

---

## Tech Stack

| Layer       | Technology                                      |
|-------------|------------------------------------------------|
| Backend     | Python, Flask, Flask-SocketIO, Flask-CORS       |
| ML          | scikit-learn, XGBoost, pandas, numpy            |
| F1 Data     | FastF1 (official F1 timing data)                |
| Frontend    | React 19, Vite, Tailwind CSS, Recharts          |
| Realtime    | Socket.IO (WebSocket)                           |
| Routing     | React Router v6                                 |
| State       | TanStack Query, React useState/useEffect        |

---

## Output Files

Standalone CLI predictions are saved to:

```
outputs/race_XX_YYYYMMDD_HHMMSS.txt
```

These contain the full prediction evolution per lap, accuracy scores, and the final classification.

---

**Project**: Final Work 2026 - Erasmus Hogeschool Brussel
