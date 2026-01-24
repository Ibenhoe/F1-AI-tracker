import './PreRaceAnalysis.css'

export default function PreRaceAnalysis() {
  return (
    <div className="pre-race-analysis">
      <h1>Pre-Race Analyse</h1>
      <div className="analysis-grid">
        <div className="analysis-card">
          <h2>Problemen</h2>
          <p>Hier komt de analyse van mogelijke problemen...</p>
        </div>
        <div className="analysis-card">
          <h2>Setup Optimalisatie</h2>
          <p>Hier komt de setup analyse...</p>
        </div>
        <div className="analysis-card">
          <h2>Voorspellingen</h2>
          <p>Hier komen de AI voorspellingen voor de race...</p>
        </div>
      </div>
    </div>
  )
}
