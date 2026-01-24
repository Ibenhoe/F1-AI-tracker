import { useState } from 'react'
import './RaceControls.css'
import apiClient from '../services/apiClient'

export default function RaceControls({ raceInitialized, raceRunning, connected }) {
  const [speed, setSpeed] = useState(1.0)

  const handleStart = () => {
    if (raceInitialized) {
      apiClient.startRace(speed)
    }
  }

  const handlePause = () => {
    apiClient.pauseRace()
  }

  const handleResume = () => {
    apiClient.resumeRace()
  }

  const handleSpeedChange = (newSpeed) => {
    setSpeed(newSpeed)
    apiClient.setSimulationSpeed(newSpeed)
  }

  return (
    <div className="race-controls">
      <div className="controls-section">
        <div className="control-group">
          <button
            className={`control-btn ${raceRunning ? 'disabled' : ''}`}
            onClick={handleStart}
            disabled={!raceInitialized || raceRunning || !connected}
          >
            ▶️ Start Race
          </button>
          
          <button
            className={`control-btn ${!raceRunning ? 'disabled' : ''}`}
            onClick={handlePause}
            disabled={!raceRunning || !connected}
          >
            ⏸️ Pause
          </button>
          
          <button
            className={`control-btn ${!raceRunning ? 'disabled' : ''}`}
            onClick={handleResume}
            disabled={raceRunning || !connected}
          >
            ▶️ Resume
          </button>
        </div>

        <div className="control-group">
          <label htmlFor="speed-control">Simulation Speed:</label>
          <select
            id="speed-control"
            value={speed}
            onChange={(e) => handleSpeedChange(parseFloat(e.target.value))}
            className="speed-select"
            disabled={!connected}
          >
            <option value={0.5}>0.5x (Slow)</option>
            <option value={1.0}>1.0x (Normal)</option>
            <option value={2.0}>2.0x (Fast)</option>
            <option value={4.0}>4.0x (Very Fast)</option>
          </select>
        </div>

        <div className="status-indicator">
          <span className={`status-dot ${connected ? 'connected' : 'disconnected'}`}></span>
          <span className="status-text">
            {!connected && 'Connecting...'}
            {connected && !raceInitialized && 'Loading race...'}
            {connected && raceInitialized && !raceRunning && 'Ready'}
            {raceRunning && 'Race Running'}
          </span>
        </div>
      </div>
    </div>
  )
}
