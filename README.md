
# F1 AI Tracker

**Final Work 2026 – Applied Computer Science**  
**Erasmushogeschool Brussel (EhB)**

---

## Overview

**F1 AI Tracker** is a real‑time Formula 1 race prediction and analysis platform that combines live race telemetry with machine learning. The system replays historical F1 races lap‑by‑lap, continuously trains a prediction model on incoming data, and visualizes predictions, race events, and analytics through an interactive web dashboard.

Unlike traditional race analytics tools, this system does **incremental machine learning during the race**, meaning predictions improve continuously as more lap data becomes available.

The goal of the project is to simulate how a **real‑time decision support system** could operate in motorsport analytics.

---

# System Architecture

The project consists of two main components:

- **Python Backend**
- **React Frontend**

These communicate in real time using **WebSockets (Socket.IO)**.

```
React Frontend  <--WebSocket-->  Flask Backend  <--->  FastF1 API
                                      |
                                      |
                               Machine Learning Models
```

---

# Backend

The backend is implemented using **Python and Flask** and is responsible for data processing, race simulation, and machine learning.

### Responsibilities

- Fetch lap‑by‑lap race data using the **FastF1 API**
- Simulate races lap‑by‑lap at configurable speeds
- Train machine learning models incrementally during the race
- Detect race events such as:
  - overtakes
  - battles
  - pit stops
- Calculate overtake probabilities
- Send real‑time updates to the frontend using **Socket.IO**

### Backend Features

- Race replay engine
- Continuous ML training
- Real‑time event generation
- Battle detection algorithm
- Tire strategy modeling
- FastF1 caching system

---

# Frontend

The frontend was built using **React (Vite)** and focuses on real‑time visualization of race data, AI predictions, and telemetry.

The interface is designed to be interactive, data‑driven, and optimized for live race analysis.

### Key Frontend Features

- **Live race dashboard**
  - driver standings
  - lap times
  - tire compounds
  - pit stop information

- **Prediction insights**
  - top‑5 predicted race finishers
  - model confidence scores

- **Race events**
  - battle detection
  - overtakes
  - pit stops
  - race notifications

- **Race replay system**
  - interactive lap‑by‑lap replay
  - track renderer visualization
  - focus mode for individual drivers

- **Data visualisation**
  - driver position history chart
  - race progress indicators
  - weather data

- **Pre‑race analysis**
  - predicted podium
  - tire strategy insights
  - driver statistics

- **Additional features**
  - dark / light theme
  - responsive layout
  - real‑time updates via WebSockets

---

# Machine Learning

The system uses two different ML approaches:

### Pre‑Race Model

Used before the race starts.

- Model: **XGBoost**
- Trained on historical race data from **2015‑2024**
- Predicts probable race outcome based on qualifying and historical performance

### Live Race Model

Used during the race simulation.

- Model: **SGDRegressor**
- Training method: `partial_fit()`
- Continuously updated after every lap

### Features Used

- grid position
- driver age
- constructor points
- circuit ID
- tire compound
- tire age
- pit stop count

To prevent unrealistic predictions, **confidence scores are capped at 85%**.

---

# Project Structure

```
F1-AI-tracker
│
├── app.py
├── race_simulator.py
├── continuous_model_learner_v2.py
├── fastf1_data_fetcher.py
├── battle_detector.py
├── event_generator.py
├── race_predictor.py
│
├── model
│   ├── prerace_model.py
│   ├── tire_strategy_model.py
│   └── tire_strategy_ml.py
│
├── data
│   ├── f1_historical_5years.csv
│   ├── processed_f1_training_data.csv
│   └── f1_weather_data.csv
│
├── models
│   ├── compound_model.json
│   ├── pit_stop_model.json
│   └── stops_model.json
│
├── frontend
│   ├── src
│   │   ├── pages
│   │   ├── components
│   │   ├── services
│   │   └── utils
│   │
│   ├── vite.config.js
│   └── package.json
│
├── analysis
├── scripts
├── cache
└── requirements.txt
```

---

# How the System Works

1. A race from the **2024 Formula 1 season** is selected in the dashboard.
2. The backend loads lap data using the **FastF1 API**.
3. The **pre‑race model** generates an initial prediction.
4. The race simulation begins and processes one lap at a time.
5. After each lap:
   - race events are detected
   - predictions are updated
   - the ML model trains using `partial_fit()`
6. Updates are sent to the frontend via **WebSockets**.
7. The frontend visualizes the data in real time.

---

# Installation

## Requirements

- Python **3.10+**
- Node.js **18+**

---

# Backend Setup

```
pip install -r requirements.txt
python app.py
```

Backend runs at:

```
http://localhost:5000
```

---

# Frontend Setup

```
cd frontend
npm install
npm run dev
```

Frontend runs at:

```
http://localhost:5173
```

---

# Available Races (2024 Season)

| Nr | Race | Nr | Race |
|---|---|---|---|
| 1 | Bahrain | 12 | United Kingdom |
| 2 | Saudi Arabia | 13 | Hungary |
| 3 | Australia | 14 | Belgium |
| 4 | Japan | 15 | Netherlands |
| 5 | China | 16 | Italy |
| 6 | Miami | 17 | Azerbaijan |
| 7 | Imola | 18 | Singapore |
| 8 | Monaco | 19 | Austin |
| 9 | Canada | 20 | Mexico |
|10 | Spain | 21 | Brazil |
|11 | Austria | 22 | Abu Dhabi |

---

# Technology Stack

| Layer | Technology |
|------|-------------|
| Backend | Python, Flask, Flask‑SocketIO |
| Machine Learning | scikit‑learn, XGBoost |
| Data | FastF1 |
| Frontend | React 19, Vite |
| Styling | Tailwind CSS |
| Charts | Recharts |
| Real‑time | Socket.IO |
| Routing | React Router |
| State | TanStack Query |

---

# Output Files

Standalone CLI predictions are stored in:

```
outputs/race_XX_YYYYMMDD_HHMMSS.txt
```

These files contain:

- prediction evolution per lap
- model accuracy scores
- final race classification

---

# Project

**Final Work – Applied Computer Science**  
**Erasmushogeschool Brussel – 2026**
