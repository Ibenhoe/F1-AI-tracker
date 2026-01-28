import './ModelMetricsPanel.css'

export default function ModelMetricsPanel({ modelMetrics, currentLap, totalLaps }) {
  if (!modelMetrics) {
    return (
      <div className="model-metrics-container">
        <div className="model-header">
          <h3>🧠 AI Model Metrics</h3>
        </div>
        <div className="model-loading">
          <p>Loading model metrics...</p>
        </div>
      </div>
    )
  }

  const maturityPercentage = Math.min(100, modelMetrics.model_maturity_percentage || 0)
  const modelStatus = modelMetrics.learning_status || 'Initializing'
  
  // Determine model health color based on updates
  const getHealthColor = (updates) => {
    if (updates < 50) return '#ff6b6b' // Red - Early
    if (updates < 100) return '#ffa500' // Orange - Learning
    if (updates < 150) return '#ffeb3b' // Yellow - Training
    return '#4caf50' // Green - Optimized
  }

  const getStatusIcon = (status) => {
    if (status.includes('Optimized')) return '✅'
    if (status.includes('Training')) return '🔄'
    return '⏳'
  }

  const healthColor = getHealthColor(modelMetrics.total_updates)

  return (
    <div className="model-metrics-container">
      <div className="model-header">
        <h3>🧠 AI Model - Real-time Learning</h3>
        <span className={`status-badge ${modelStatus.toLowerCase().replace(' ', '-')}`}>
          {getStatusIcon(modelStatus)} {modelStatus}
        </span>
      </div>

      {/* Model Overview */}
      <div className="model-overview">
        <div className="overview-item">
          <span className="label">Model Type</span>
          <span className="value">{modelMetrics.model_type}</span>
        </div>
        <div className="overview-item">
          <span className="label">Confidence Cap</span>
          <span className="value">{modelMetrics.confidence_cap}</span>
        </div>
        <div className="overview-item">
          <span className="label">Pre-trained</span>
          <span className="value">
            {modelMetrics.is_pretrained ? '✅ Yes' : '❌ No'}
          </span>
        </div>
      </div>

      {/* Model Updates Progress */}
      <div className="updates-section">
        <div className="section-title">Model Updates</div>
        <div className="update-stats">
          <div className="stat-item">
            <span className="stat-label">Total Updates:</span>
            <span className="stat-value" style={{ color: healthColor }}>
              {modelMetrics.total_updates}
            </span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Samples Processed:</span>
            <span className="stat-value">
              {modelMetrics.samples_processed}
            </span>
          </div>
        </div>

        {/* Maturity Progress Bar */}
        <div className="maturity-bar-container">
          <label>Model Maturity</label>
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{
                width: `${maturityPercentage}%`,
                backgroundColor: healthColor,
              }}
            />
          </div>
          <span className="progress-label">{Math.round(maturityPercentage)}%</span>
        </div>
      </div>

      {/* Models Status */}
      <div className="models-status-section">
        <div className="section-title">Model Components</div>
        <div className="components-list">
          <div className={`component ${modelMetrics.sgd_model_ready ? 'active' : 'inactive'}`}>
            <span className="icon">📊</span>
            <span className="name">SGDRegressor</span>
            <span className="status">
              {modelMetrics.sgd_model_ready ? '✅' : '⏳'}
            </span>
          </div>
          <div className={`component ${modelMetrics.mlp_model_ready ? 'active' : 'inactive'}`}>
            <span className="icon">🧬</span>
            <span className="name">MLPRegressor</span>
            <span className="status">
              {modelMetrics.mlp_model_ready ? '✅' : '⏳'}
            </span>
          </div>
          <div className={`component ${modelMetrics.rf_classifier_ready ? 'active' : 'inactive'}`}>
            <span className="icon">🌲</span>
            <span className="name">RandomForest</span>
            <span className="status">
              {modelMetrics.rf_classifier_ready ? '✅' : '⏳'}
            </span>
          </div>
        </div>
      </div>

      {/* Features Used */}
      {modelMetrics.feature_names && modelMetrics.feature_names.length > 0 && (
        <div className="features-section">
          <div className="section-title">Features Used</div>
          <div className="features-list">
            {modelMetrics.feature_names.map((feature, idx) => (
              <span key={idx} className="feature-tag">
                {feature}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Recent Performance */}
      {modelMetrics.recent_mae_average !== undefined && (
        <div className="performance-section">
          <div className="section-title">Recent Performance</div>
          <div className="perf-items">
            <div className="perf-item">
              <span className="label">Avg MAE:</span>
              <span className="value">
                {modelMetrics.recent_mae_average.toFixed(2)}
              </span>
            </div>
            <div className="perf-item">
              <span className="label">Trend:</span>
              <span className="value">
                {modelMetrics.mae_trend === 'improving' ? '📈 Improving' : '→ Stable'}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Learning Notes */}
      <div className="notes-section">
        <p className="note-text">
          ℹ️ The AI model learns <strong>continuously per lap</strong> using incremental learning (SGDRegressor partial_fit).
          Predictions are <strong>probabilistic and realistic</strong> - never showing 100% certainty.
        </p>
      </div>
    </div>
  )
}
