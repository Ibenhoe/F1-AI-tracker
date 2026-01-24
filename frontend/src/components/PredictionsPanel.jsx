import './PredictionsPanel.css'

export default function PredictionsPanel({ predictions, currentLap }) {
  const mockPredictions = [
    { position: 1, driver: 'Verstappen', confidence: 92, change: 'stable' },
    { position: 2, driver: 'Piastri', confidence: 88, change: 'up' },
    { position: 3, driver: 'Norris', confidence: 85, change: 'down' },
    { position: 4, driver: 'Leclerc', confidence: 80, change: 'stable' },
  ]

  return (
    <div className="predictions-container">
      <div className="predictions-header">
        <h2>🤖 AI Voorspellingen</h2>
        <span className="lap-indicator">Ronde: {currentLap}</span>
      </div>
      <div className="predictions-list">
        {mockPredictions.map((pred) => (
          <div key={pred.position} className="prediction-row">
            <div className="pred-position">{pred.position}</div>
            <div className="pred-info">
              <div className="pred-driver">{pred.driver}</div>
              <div className="pred-confidence">
                <div className="confidence-bar">
                  <div 
                    className="confidence-fill"
                    style={{ width: `${pred.confidence}%` }}
                  ></div>
                </div>
                <span className="confidence-text">{pred.confidence}%</span>
              </div>
            </div>
            <div className={`pred-change ${pred.change}`}>
              {pred.change === 'up' && '📈'}
              {pred.change === 'down' && '📉'}
              {pred.change === 'stable' && '→'}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
