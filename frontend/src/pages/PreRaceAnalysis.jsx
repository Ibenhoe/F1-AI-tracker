import './PreRaceAnalysis.css'
import { useState, useEffect, useRef } from 'react'
import RaceSelector from '../components/RaceSelector'
import Card from '../components/ui/Card'

export default function PreRaceAnalysis() {
  const [raceNumber, setRaceNumber] = useState(21)
  const [predictions, setPredictions] = useState([])
  const [tireStrategies, setTireStrategies] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [analysis, setAnalysis] = useState(null)
  const [circuitAnalysis, setCircuitAnalysis] = useState(null)
  const retryTimerRef = useRef(null)

  // Fetch both pre-race analysis and tire strategy when race changes
  useEffect(() => {
    if (raceNumber) {
      fetchPreRaceData(raceNumber)
    }
    
    // Cleanup retry timer on unmount
    return () => {
      if (retryTimerRef.current) {
        clearTimeout(retryTimerRef.current)
      }
    }
  }, [raceNumber])

  const fetchPreRaceData = async (raceNum) => {
    try {
      setLoading(true)
      setError(null)
      
      // Fetch both APIs in parallel
      const [predResponse, tireResponse] = await Promise.all([
        fetch('http://localhost:5000/api/race/prerace-analysis', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ race_number: raceNum, grid: null })
        }),
        fetch('http://localhost:5000/api/race/tire-strategy', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ race_number: raceNum })
        })
      ])

      if (!predResponse.ok) {
        throw new Error('Failed to fetch pre-race analysis')
      }

      const predData = await predResponse.json()
      if (predData.predictions) {
        setPredictions(predData.predictions)
        setAnalysis(predData.analysis)
      }

      // Tire strategy might fail gracefully or be loading (202 = pending)
      if (tireResponse.ok || tireResponse.status === 202) {
        const tireData = await tireResponse.json()
        if (tireData.strategies && tireData.strategies.length > 0) {
          setTireStrategies(tireData.strategies)
          setCircuitAnalysis(tireData.circuit_analysis)
        }
        if (tireResponse.status === 202) {
          // Model is still loading, retry after 2 seconds
          console.log('Tire strategy model is loading, will retry in 2 seconds...')
          if (retryTimerRef.current) clearTimeout(retryTimerRef.current)
          retryTimerRef.current = setTimeout(() => {
            fetchPreRaceData(raceNum)
          }, 2000)
        }
      } else if (tireResponse.status === 503) {
        // Model temporarily unavailable
        console.warn('Tire strategy model temporarily unavailable, skipping...')
      }
    } catch (err) {
      console.error('Error fetching pre-race data:', err)
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
    <div className="w-full min-h-screen bg-white dark:bg-neutral-950 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-neutral-900 dark:text-neutral-100 mb-2">
            Pre-Race AI Analysis
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400">
            Voorspellingen en bandenstrategie voor alle 21 F1 races
          </p>
        </div>

        {/* Race Selector */}
        <Card>
          <div className="flex flex-col gap-4">
            <h2 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
              Kies een race
            </h2>
            <RaceSelector 
              selectedRace={raceNumber}
              onSelectRace={handleRaceSelect} 
            />
          </div>
        </Card>

        {/* Error State */}
        {error && (
          <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
            <p className="text-red-700 dark:text-red-300">Fout: {error}</p>
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="flex flex-col items-center justify-center py-16 gap-4">
            <div className="w-12 h-12 border-4 border-neutral-200 dark:border-neutral-700 border-t-neutral-900 dark:border-t-neutral-100 rounded-full animate-spin" />
            <p className="text-neutral-600 dark:text-neutral-400">AI-model laden en voorspellingen genereren...</p>
          </div>
        )}

        {/* Main Content */}
        {!loading && predictions.length > 0 && (
          <div className="space-y-6">
            {/* AI Predictions Panel */}
            <Card>
              <div className="flex flex-col gap-6">
                <div>
                  <h2 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                    🏁 AI-Voorspellingen
                  </h2>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-neutral-200 dark:border-neutral-800">
                        <th className="text-left py-3 px-4 font-semibold text-neutral-700 dark:text-neutral-300">#</th>
                        <th className="text-left py-3 px-4 font-semibold text-neutral-700 dark:text-neutral-300">Driver</th>
                        <th className="text-left py-3 px-4 font-semibold text-neutral-700 dark:text-neutral-300">Team</th>
                        <th className="text-left py-3 px-4 font-semibold text-neutral-700 dark:text-neutral-300">Grid</th>
                        <th className="text-left py-3 px-4 font-semibold text-neutral-700 dark:text-neutral-300">Score</th>
                        <th className="text-left py-3 px-4 font-semibold text-neutral-700 dark:text-neutral-300">Confidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {predictions.slice(0, 10).map((pred, idx) => (
                        <tr key={idx} className="border-b border-neutral-100 dark:border-neutral-800 hover:bg-neutral-50 dark:hover:bg-neutral-800/50 transition">
                          <td className="py-3 px-4 font-semibold text-neutral-900 dark:text-neutral-100">{idx + 1}</td>
                          <td className="py-3 px-4">
                            <div className="font-semibold text-neutral-900 dark:text-neutral-100">{pred.driver_name || pred.driver}</div>
                            <div className="text-xs text-neutral-600 dark:text-neutral-400">#{pred.number}</div>
                          </td>
                          <td className="py-3 px-4">
                            <span className="inline-block px-2 py-1 bg-neutral-100 dark:bg-neutral-800 text-neutral-700 dark:text-neutral-300 text-xs font-medium rounded">
                              {pred.team}
                            </span>
                          </td>
                          <td className="py-3 px-4">
                            <span className="font-semibold text-neutral-900 dark:text-neutral-100">P{pred.grid_position}</span>
                          </td>
                          <td className="py-3 px-4 font-semibold text-neutral-900 dark:text-neutral-100">
                            {pred.ai_score.toFixed(1)}
                          </td>
                          <td className="py-3 px-4">
                            <div className="flex items-center gap-2">
                              <div className="w-16 h-2 bg-neutral-200 dark:bg-neutral-700 rounded-full overflow-hidden">
                                <div
                                  className="h-full transition-all"
                                  style={{
                                    width: `${pred.confidence}%`,
                                    backgroundColor: getConfidenceColor(pred.confidence)
                                  }}
                                />
                              </div>
                              <span className="w-12 text-right font-semibold text-neutral-900 dark:text-neutral-100">
                                {pred.confidence.toFixed(0)}%
                              </span>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </Card>

            {/* 🔍 DRIVERS TO WATCH - ANOMALIES DETECTION */}
            {predictions.some(p => p.anomaly) && (
              <Card>
                <div className="flex flex-col gap-6">
                  <div>
                    <h2 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 flex items-center gap-2">
                      👀 Drivers to Watch
                    </h2>
                    <p className="text-sm text-neutral-600 dark:text-neutral-400 mt-1">
                      Onverwachte prestaties in qualifying die kunnen beïnvloeden hoe de race verloopt
                    </p>
                  </div>

                  <div className="grid gap-3">
                    {predictions
                      .filter(p => p.anomaly)
                      .map((pred, idx) => {
                        const anom = pred.anomaly
                        const severityColors = {
                          1: 'border-yellow-200 dark:border-yellow-900 bg-yellow-50 dark:bg-yellow-900/20',
                          2: 'border-orange-200 dark:border-orange-900 bg-orange-50 dark:bg-orange-900/20',
                          3: 'border-red-200 dark:border-red-900 bg-red-50 dark:bg-red-900/20'
                        }
                        const textColor = {
                          1: 'text-yellow-800 dark:text-yellow-200',
                          2: 'text-orange-800 dark:text-orange-200',
                          3: 'text-red-800 dark:text-red-200'
                        }
                        
                        return (
                          <div
                            key={idx}
                            className={`border-l-4 rounded-lg p-4 transition ${severityColors[anom.severity] || severityColors[1]}`}
                          >
                            <div className={`font-semibold ${textColor[anom.severity] || textColor[1]} mb-2`}>
                              {anom.message}
                            </div>
                            <div className={`text-sm ${textColor[anom.severity] || textColor[1]}`}>
                              💡 {anom.explanation}
                            </div>
                            <div className={`text-xs mt-2 ${textColor[anom.severity] || textColor[1]} opacity-75`}>
                              [{anom.type}] Severity: Level {anom.severity}/3
                            </div>
                          </div>
                        )
                      })}
                  </div>
                </div>
              </Card>
            )}

            {/* Tire Strategy Panel - GENERAL STRATEGIES (1-2) */}
            {tireStrategies.length > 0 && (
              <Card className="xl:col-span-2">
                <div className="flex flex-col gap-4">
                  <div>
                    <h2 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-1">
                      🛞 Bandenstrategie
                    </h2>
                    {circuitAnalysis && (
                      <p className="text-sm text-neutral-600 dark:text-neutral-400">
                        Circuit wear: {(circuitAnalysis.tire_wear_rate * 100).toFixed(0)}% | Type: {circuitAnalysis.track_type}
                      </p>
                    )}
                  </div>

                  <div className="space-y-3">
                    {tireStrategies.map((strat, idx) => (
                      <div key={idx} className="border rounded-lg border-neutral-200 dark:border-neutral-800 p-4">
                        <div className="flex items-start justify-between mb-3">
                          <div>
                            <div className="flex items-center gap-2 mb-1">
                              <span className="inline-block px-2 py-1 bg-neutral-100 dark:bg-neutral-800 text-neutral-700 dark:text-neutral-300 text-xs font-semibold rounded">
                                {strat.rank === 1 ? '✓ PRIMAIR' : '⚡ ALT'}
                              </span>
                              <h3 className="text-base font-semibold text-neutral-900 dark:text-neutral-100">
                                {strat.strategy_type.replace(/_/g, ' ').toUpperCase()}
                              </h3>
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="text-lg font-bold text-emerald-600 dark:text-emerald-400">
                              {strat.confidence.toFixed(0)}%
                            </div>
                            <div className="text-xs text-neutral-600 dark:text-neutral-400">
                              betrouwbaarheid
                            </div>
                          </div>
                        </div>

                        <div className="grid grid-cols-2 gap-3 text-sm">
                          <div>
                            <span className="block text-neutral-600 dark:text-neutral-400 text-xs font-medium mb-1">
                              Pit Stops:
                            </span>
                            <span className="text-neutral-900 dark:text-neutral-100 font-medium">
                              {strat.pit_stop_laps.length === 0 
                                ? 'Geen' 
                                : strat.pit_stop_laps.map((lap, i) => (
                                    <span key={i}>
                                      Lap {lap}{i < strat.pit_stop_laps.length - 1 ? ', ' : ''}
                                    </span>
                                  ))
                              }
                            </span>
                          </div>

                          <div>
                            <span className="block text-neutral-600 dark:text-neutral-400 text-xs font-medium mb-1">
                              Bandenkeuze:
                            </span>
                            <div className="flex items-center gap-1 flex-wrap">
                              {strat.tire_sequence.map((compound, i) => (
                                <span key={i} className="inline-flex items-center gap-1">
                                  <span className={`px-2 py-1 rounded text-xs font-semibold 
                                    ${compound === 'SOFT' ? 'bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-200' : ''}
                                    ${compound === 'MEDIUM' ? 'bg-yellow-100 dark:bg-yellow-900 text-yellow-700 dark:text-yellow-200' : ''}
                                    ${compound === 'HARD' ? 'bg-white dark:bg-neutral-700 text-neutral-700 dark:text-neutral-200' : ''}
                                  `}>
                                    {compound}
                                  </span>
                                  {i < strat.tire_sequence.length - 1 && <span className="text-neutral-400">→</span>}
                                </span>
                              ))}
                            </div>
                          </div>
                        </div>

                        {strat.recommendation && (
                          <div className="mt-3 pt-3 border-t border-neutral-200 dark:border-neutral-800">
                            <p className="text-sm text-neutral-700 dark:text-neutral-300">
                              💡 {strat.recommendation}
                            </p>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              </Card>
            )}

          </div>
        )}

        {!loading && predictions.length === 0 && !error && (
          <Card>
            <div className="text-center py-16">
              <p className="text-neutral-600 dark:text-neutral-400 text-lg">
                👉 Selecteer een race om AI-voorspellingen te zien
              </p>
            </div>
          </Card>
        )}
      </div>
    </div>
  )
}

