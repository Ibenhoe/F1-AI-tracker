# F1 AI Tracker - Backend Server

Flask + SocketIO backend voor real-time F1 race simulation met WebSocket communicatie.

## Features

- **REST API** voor race data en initializzatie
- **WebSocket (SocketIO)** voor real-time lap updates
- **Race Simulator** die lap-by-lap simulaties uitvoert
- **FastF1 Integration** voor echte race data
- **AI Model Integration** voor real-time voorspellingen

## Installation

### Python Environment

```bash
# Requirements al geinstalleerd
pip install -r ../requirements.txt
```

### Starten van de Backend Server

```bash
python app.py
```

Server draait op: `http://localhost:5000`

## API Endpoints

### REST API

- `GET /api/health` - Health check
- `GET /api/races` - Beschikbare races (1-21)
- `POST /api/race/init` - Race initialiseren
  - Body: `{"race_number": 21}`
  - Response: `{race_name, total_laps, drivers}`

## WebSocket Events

### Client → Server (Emit)

- `race/start` - Race starten met snelheid
  - Payload: `{speed: 1.0}`
- `race/pause` - Race pauzeren
- `race/resume` - Race hervatten
- `race/speed` - Simulatie snelheid wijzigen
  - Payload: `{speed: 2.0}`

### Server → Client (Listen)

- `connect_response` - Verbinding bevestiging
- `race/started` - Race gestart
  - Payload: `{current_lap, total_laps}`
- `lap/update` - Lap update (per lap uitgestuurd)
  - Payload: `{lap_number, drivers[], predictions[], events[]}`
- `race/paused` - Race gepauzeerd
- `race/resumed` - Race hervatten
- `race/finished` - Race beëindigd
  - Payload: `{final_standings}`
- `race/error` - Error opgetreden
  - Payload: `{error}`

## Driver Data Structure

Elk driver object bevat:

```javascript
{
  position: 1,
  driver_code: "VER",
  driver_name: "Max Verstappen",
  team: "Red Bull Racing",
  lap_time: "1:28.473",
  tire_compound: "SOFT",      // SOFT, MEDIUM, HARD
  tire_age: 5,                // Laps op deze banden
  pit_stops: 1,               // Aantal pitstops
  gap: "+0.000",              // Gap naar P1
  laps_completed: 58,         // Totaal afgelegde rondes
  position_change: 0,         // Positieverandering deze ronde
  dnf: false                  // Did Not Finish
}
```

## Prediction Data Structure

```javascript
{
  position: 1,
  driver_code: "VER",
  confidence: 95,             // 0-100%
  trend: "stable"             // up, down, stable
}
```

## Race Simulator

De `RaceSimulator` class simuleert lap-by-lap:

- Haalt real data op via FastF1
- Voert AI model voorspellingen uit
- Update driver states (positie, banden, pit stops)
- Genereert race events (pit stops, crashes, etc.)

## Voorbeeld Initialisatie

```python
from app import app, socketio

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
```

## Architecture

```
app.py (Flask + SocketIO)
  ├── REST API Endpoints
  ├── WebSocket Handlers
  └── Background Simulation Thread
        └── race_simulator.py (RaceSimulator)
              ├── FastF1 Data Fetcher
              ├── AI Model (ContinuousModelLearner)
              └── Driver State Management
```

## Troubleshooting

- **Connection timeout**: Controleer of backend op localhost:5000 draait
- **No lap data**: Race-nummer mogelijk niet beschikbaar in FastF1
- **Model not loading**: ContinuousModelLearner model.pkl moet bestaan
