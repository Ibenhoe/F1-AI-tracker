import './RaceInfo.css'

export default function RaceInfo({ data }) {
  if (!data) {
    return <div className="race-info loading">Race info laden...</div>
  }

  return (
    <div className="race-info">
      <div className="race-header">
        <h1>{data.race}</h1>
      </div>
      <div className="race-stats">
        <div className="race-stat-item">
          <div className="race-stat-label">Huidige Ronde</div>
          <div className="race-stat-value">{data.currentLap} / {data.totalLaps}</div>
        </div>
        <div className="race-progress">
          <div className="progress-bar">
            <div 
              className="progress-fill"
              style={{ width: `${(data.currentLap / data.totalLaps) * 100}%` }}
            ></div>
          </div>
          <div className="progress-text">
            {Math.round((data.currentLap / data.totalLaps) * 100)}% voortgang
          </div>
        </div>
      </div>
    </div>
  )
}
