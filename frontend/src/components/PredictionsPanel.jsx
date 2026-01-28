import './PredictionsPanel.css'

export default function PredictionsPanel({ predictions, currentLap, modelMetrics, totalLaps }) {
  // Use real predictions if available, fallback to empty
  const displayPredictions = predictions && predictions.length > 0 ? predictions : []

  // Calculate model status
  const getModelStatus = () => {
    if (!modelMetrics) return { status: 'Initializing...', icon: '⏳', color: '#ff6b6b' }
    
    const maturity = modelMetrics.model_maturity_percentage || 0
    if (maturity >= 100) return { status: 'Optimized', icon: '✅', color: '#4caf50' }
    if (maturity >= 50) return { status: 'Training', icon: '🔄', color: '#ffeb3b' }
    if (maturity >= 25) return { status: 'Learning', icon: '🔄', color: '#ffa500' }
    return { status: 'Initializing...', icon: '⏳', color: '#ff6b6b' }
  }

  const modelStatus = getModelStatus()

  if (displayPredictions.length === 0) {
    return (
      <div className="predictions-container">
        <div className="predictions-header">
          <div>
            <h2>🤖 Race Voorspelling</h2>
            {modelMetrics && (
              <div className="model-status-badge" style={{ borderColor: modelStatus.color, color: modelStatus.color }}>
                {modelStatus.icon} Model: {modelStatus.status} | Updates: {modelMetrics.total_updates}
              </div>
            )}
          </div>
          <span className="lap-indicator">Ronde: {currentLap}</span>
        </div>
        <div className="predictions-list">
          <div className="no-predictions">
            AI model wordt getraind... ⏳
            {modelMetrics && (
              <div className="model-training-info">
                Maturity: {Math.round(modelMetrics.model_maturity_percentage || 0)}%
              </div>
            )}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="predictions-container">
      <div className="predictions-header">
        <div>
          <h2>🤖 Race Voorspelling (Top 5)</h2>
          {modelMetrics && (
            <div className="model-status-inline">
              <div className="status-info">
                <span className="status-badge" style={{ backgroundColor: modelStatus.color }}>
                  {modelStatus.icon} {modelStatus.status}
                </span>
                <span className="updates-count">Updates: {modelMetrics.total_updates}</span>
                <span className="maturity-info">Maturity: {Math.round(modelMetrics.model_maturity_percentage || 0)}%</span>
              </div>
              {modelMetrics.sgd_model_ready || modelMetrics.mlp_model_ready || modelMetrics.rf_classifier_ready ? (
                <div className="components-mini">
                  {modelMetrics.sgd_model_ready && <span className="component-badge">SGD ✅</span>}
                  {modelMetrics.mlp_model_ready && <span className="component-badge">MLP ✅</span>}
                  {modelMetrics.rf_classifier_ready && <span className="component-badge">RF ✅</span>}
                </div>
              ) : null}
            </div>
          )}
        </div>
        <span className="lap-indicator">Ronde: {currentLap}</span>
      </div>
      
      <div className="predictions-list">
        {displayPredictions.map((pred, idx) => (
          <div key={idx} className="prediction-row">
            <div className="pred-position">
              <span className="pred-rank">#{idx + 1}</span>
              <span className="pred-change">{pred.trend === 'up' ? '📈' : (pred.trend === 'down' ? '📉' : '→')}</span>
            </div>
            <div className="pred-info">
              <div className="pred-driver">
                {pred.driver_name || pred.driver_code}
                <span className="pred-pos-change">
                  Pos: {Math.round(pred.position)} → {Math.round(pred.prediction)}
                </span>
              </div>
              <div className="pred-strategy">
                <span className="pred-grid">Start: P{pred.grid_pos}</span>
                <span className="pred-pits">Pit stops: {pred.pit_stops}</span>
              </div>
              <div className="pred-confidence">
                <div className="confidence-bar">
                  <div 
                    className="confidence-fill"
                    style={{ width: `${pred.confidence}%` }}
                  ></div>
                </div>
                <span className="confidence-text">{Math.round(pred.confidence)}%</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {modelMetrics && modelMetrics.recent_mae_average !== undefined && (
        <div className="model-performance-footer">
          <span className="perf-label">Model Performance:</span>
          <span className="perf-mae">MAE: {modelMetrics.recent_mae_average.toFixed(2)}</span>
          <span className="perf-trend">{modelMetrics.mae_trend === 'improving' ? '📈' : '→'} {modelMetrics.mae_trend}</span>
        </div>
      )}
    </div>
  )
}
