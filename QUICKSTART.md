# F1 AI Tracker - Setup & Launch Guide

## Quick Start

### 1. Backend Starten (Python Server)

```bash
cd F1-AI-tracker
python app.py
```

Server start op: `http://localhost:5000`

Output:
```
[BACKEND] Starting F1 AI Tracker Backend...
[BACKEND] Server running on http://localhost:5000
```

### 2. Frontend Starten (React Dev Server)

In een NIEUWE terminal:

```bash
cd F1-AI-tracker/frontend
npm run dev
```

Frontend draait op: `http://localhost:5173`

### 3. Browser Openen

Open http://localhost:5173 in je browser

## Usage

1. **Verbinding**: Dashboard connecteert automatisch met backend
2. **Race Selecteren**: Abu Dhabi (race 21) is default geladen
3. **Simulatie Starten**: Klik "▶️ Start Race"
4. **Snelheid Aanpassen**: Selecteer snelheid (0.5x - 4.0x)
5. **Volgen**: Race updates lap-by-lap in real-time

## Data Weergegeven

### Klassering Kolom (Links)
- Driver positie & naam
- Huidige ronde tijd
- Banden type (SOFT/MEDIUM/HARD) + aantal laps
- Pit stops teller
- Positie verandering deze ronde

### Voorspellingen Kolom (Midden)
- AI model confidence per driver (0-100%)
- Trend indicator (📈📉→)
- Positie voorspelling

### Notificaties Kolom (Rechts)
- Pit stops
- Race events
- Simulatie status

## Architecture

```
┌─────────────────────────────────────┐
│   React Frontend (Vite)             │
│   - Dashboard                       │
│   - DriversList (met banden info)  │
│   - PredictionsPanel (AI)          │
│   - RaceControls                    │
└──────────────┬──────────────────────┘
               │ WebSocket (SocketIO)
               │
┌──────────────▼──────────────────────┐
│   Flask + SocketIO Backend          │
│   - REST API endpoints              │
│   - WebSocket handlers              │
│   - Background simulation           │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│   Race Simulator                    │
│   - FastF1 data fetcher            │
│   - AI model (XGBoost)             │
│   - Lap-by-lap simulation          │
└─────────────────────────────────────┘
```

## Files Overzicht

### Backend Files (Python)
- `app.py` - Flask server + SocketIO
- `race_simulator.py` - Lap-by-lap race simulation
- `requirements.txt` - Python dependencies

### Frontend Files (React)
- `frontend/src/App.jsx` - Hoofd app
- `frontend/src/pages/Dashboard.jsx` - Dashboard page
- `frontend/src/components/`:
  - `DriversList.jsx` - Klassering met banden/pit stops
  - `PredictionsPanel.jsx` - AI voorspellingen
  - `RaceControls.jsx` - Start/pause/speed controls
  - `WeatherWidget.jsx` - Weer info
  - `RaceInfo.jsx` - Race progress
  - `Sidebar.jsx` - Navigatie
- `frontend/src/services/apiClient.js` - API/WebSocket client
- `frontend/.env` - Backend URL config

## Troubleshooting

### Frontend connecteert niet
- Zorg dat backend draait op port 5000
- Check `.env` file: `VITE_BACKEND_URL=http://localhost:5000`
- Browser console: Ctrl+Shift+I → Console tab

### Geen race data
- Race 21 (Abu Dhabi) moet in FastF1 beschikbaar zijn
- Controleer internet verbinding (FastF1 haalt live data)
- Check backend logs voor errors

### Race simulation loopt vast
- Zorg dat AI model geladen is
- Check terminal voor Python errors
- Probeer andere race-nummer

## Next Steps

Volgende uitbreidingen:
1. Live data push van echte races
2. Video feeds integreren
3. Real-time pit stop strategy
4. Wet weather simulation
5. Accident/DNF handling
