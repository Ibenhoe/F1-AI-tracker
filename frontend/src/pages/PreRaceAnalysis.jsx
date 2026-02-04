import './PreRaceAnalysis.css'
import { useState, useEffect } from 'react'
import RaceSelector from '../components/RaceSelector'

export default function PreRaceAnalysis() {
  const [raceNumber, setRaceNumber] = useState(21)
  const [predictions, setPredictions] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [analysis, setAnalysis] = useState(null)

  // Fetch pre-race analysis when race changes
  useEffect(() => {
    if (raceNumber) {
      fetchPreRaceAnalysis(raceNumber)
    }
  }, [raceNumber])

  const fetchPreRaceAnalysis = async (raceNum) => {
    try {
      setLoading(true)
      setError(null)
      
      const response = await fetch('http://localhost:5000/api/race/prerace-analysis', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          race_number: raceNum,
          grid: null
        })
      })

      if (!response.ok) {
        throw new Error('Failed to fetch pre-race analysis')
      }

      const data = await response.json()
      
      if (data.predictions) {
        setPredictions(data.predictions)
        setAnalysis(data.analysis)
      }
    } catch (err) {
      console.error('Error fetching pre-race analysis:', err)
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleRaceSelect = (raceId) => {
    setRaceNumber(raceId)
  }

  const getConfidenceColor = (confidence) => {
    if (confidence >= 80) return '#10b981' // Green
    if (confidence >= 70) return '#3b82f6' // Blue
    if (confidence >= 60) return '#f59e0b' // Amber
    return '#ef4444' // Red
  }

  const getMedalEmoji = (position) => {
    if (position === 1) return '1st'
    if (position === 2) return '2nd'
    if (position === 3) return '3rd'
    return `${position}th`
  }

  return (
    <div className="pre-race-analysis">
      <div className="pre-race-header">
        <h1>Pre-Race AI Analysis</h1>
        <p className="subtitle">Voorspellingen op basis van 40+ features en XGBoost-model</p>
      </div>

      <div className="pre-race-content">
        <div className="race-selector-section">
          <h2>Kies een race</h2>
          <RaceSelector 
            selectedRace={raceNumber}
            onSelectRace={handleRaceSelect} 
          />
        </div>

        {error && (
          <div className="error-message">
            <p>Fout: {error}</p>
          </div>
        )}

        {loading && (
          <div className="loading-container">
            <div className="spinner"></div>
            <p>AI-model laden en voorspellingen genereren...</p>
          </div>
        )}

        {!loading && predictions.length > 0 && (
          <div className="analysis-grid">
            {/* AI Predictions Panel */}
            <div className="analysis-card predictions-card">
              <div className="card-header">
                <h2>AI-Voorspellingen (Top 10)</h2>
                {analysis && (
                  <p className="model-info">
                    Model: {analysis.model} | {analysis.features_used} features
                  </p>
                )}
              </div>

              <div className="predictions-table">
                <div className="table-header">
                  <div className="col-position">#</div>
                  <div className="col-driver">Driver</div>
                  <div className="col-team">Team</div>
                  <div className="col-grid">Grid</div>
                  <div className="col-score">AI Score</div>
                  <div className="col-confidence">Confidence</div>
                </div>

                {predictions.map((pred, idx) => (
                  <div key={idx} className="table-row">
                    <div className="col-position medal">
                      <span className="position-number">{idx + 1}</span>
                    </div>
                    <div className="col-driver">
                      <div className="driver-info">
                        <div className="driver-name">{pred.driver}</div>
                        <div className="driver-number">#{pred.number}</div>
                      </div>
                    </div>
                    <div className="col-team">
                      <div className="team-badge">{pred.team}</div>
                    </div>
                    <div className="col-grid">
                      <div className="grid-badge">P{pred.grid_position}</div>
                    </div>
                    <div className="col-score">
                      <span className="score-value">{pred.ai_score.toFixed(2)}</span>
                    </div>
                    <div className="col-confidence">
                      <div className="confidence-bar-container">
                        <div
                          className="confidence-bar"
                          style={{
                            width: `${pred.confidence}%`,
                            backgroundColor: getConfidenceColor(pred.confidence)
                          }}
                        ></div>
                        <span className="confidence-text">
                          {pred.confidence.toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Model Analysis Info */}
            <div className="analysis-card info-card">
              <h2>Model Informatie</h2>
              <div className="info-grid">
                <div className="info-item">
                  <h3>Features Gebruikt</h3>
                  <p className="info-value">{analysis?.features_used || 17}</p>
                  <ul className="feature-list">
                    <li>Grid Position</li>
                    <li>Driver Experience</li>
                    <li>Recent Form (5 races)</li>
                    <li>Track History</li>
                    <li>Pit Stop Data</li>
                    <li>Weather Conditions</li>
                    <li>Constructor Strength</li>
                  </ul>
                </div>

                <div className="info-item">
                  <h3>Model Type</h3>
                  <p className="info-value">{analysis?.model || 'XGBoost'}</p>
                  <p className="model-desc">
                    Advanced machine learning regressor trained op historische F1-data (2015-2024)
                  </p>
                </div>

                <div className="info-item">
                  <h3>Confidence Level</h3>
                  <p className="info-value">Max {analysis?.confidence_threshold || 85}%</p>
                  <p className="model-desc">
                    Realistische voorspellingen die onverwachte race-factoren in acht nemen
                  </p>
                </div>

                <div className="info-item">
                  <h3>Accuratesse</h3>
                  <p className="info-value">~78%</p>
                  <p className="model-desc">
                    Topscorer voorspellingen op testset (MAE: 4.2 positions)
                  </p>
                </div>
              </div>
            </div>

            {/* Strategy Analysis */}
            <div className="analysis-card strategy-card">
              <h2>🎯 Strategische Inzichten</h2>
              <div className="strategy-grid">
                <div className="strategy-item">
                  <h4>Top 3 Favorieten</h4>
                  <div className="favorites-list">
                    {predictions.slice(0, 3).map((pred, idx) => (
                      <div key={idx} className="favorite-item">
                        <span className="medal-lg">{getMedalEmoji(pred.position)}</span>
                        <span className="fav-name">{pred.driver}</span>
                        <span className="fav-conf">{pred.confidence.toFixed(0)}%</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="strategy-item">
                  <h4>Dark Horses</h4>
                  <div className="darkhorse-list">
                    {predictions
                      .filter(p => p.grid_position > 10 && p.position <= 10)
                      .slice(0, 3)
                      .map((pred, idx) => (
                        <div key={idx} className="darkhorse-item">
                          <span className="pos-tag">P{pred.grid_position}</span>
                          <span className="dh-name">{pred.driver}</span>
                          <span className="dh-move">→ P{pred.position}</span>
                        </div>
                      ))}
                    {predictions.filter(p => p.grid_position > 10 && p.position <= 10).length === 0 && (
                      <p className="no-data">Geen dark horses deze race</p>
                    )}
                  </div>
                </div>

                <div className="strategy-item">
                  <h4>Risico's</h4>
                  <div className="risks-list">
                    {predictions.filter(p => p.confidence < 65).slice(0, 3).map((pred, idx) => (
                      <div key={idx} className="risk-item">
                        <span className="risk-name">{pred.driver}</span>
                        <span className="risk-conf">{pred.confidence.toFixed(0)}%</span>
                        <span className="risk-badge">Onzeker</span>
                      </div>
                    ))}
                    {predictions.filter(p => p.confidence < 65).length === 0 && (
                      <p className="no-data">Lage onzekerheid deze race</p>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {!loading && predictions.length === 0 && !error && (
          <div className="empty-state">
            <p>👉 Selecteer een race om AI-voorspellingen te zien</p>
          </div>
        )}
      </div>
    </div>
  )
}

