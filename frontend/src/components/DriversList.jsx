import './DriversList.css'

export default function DriversList({ drivers, currentLap }) {
  const getTireColor = (compound) => {
    switch(compound) {
      case 'SOFT': return '#ff1744'
      case 'MEDIUM': return '#ffd600'
      case 'HARD': return '#ffffff'
      default: return '#9ca3af'
    }
  }

  return (
    <div className="drivers-list-container">
      <div className="drivers-header">
        <h2>🏎️ Klassering</h2>
        <span className="lap-indicator">Ronde: {currentLap || 0}</span>
      </div>
      <div className="drivers-list">
        {drivers && drivers.length > 0 ? (
          drivers.map((driver) => (
            <div key={driver.driver_code || driver.position} className="driver-row">
              <div className="driver-position">{driver.position}</div>
              <div className="driver-info">
                <div className="driver-name">{driver.driver_name || driver.driver_code}</div>
                <div className="driver-team">{driver.team}</div>
              </div>
              <div className="driver-data">
                <div className="data-item">
                  <span className="data-label">Ronde</span>
                  <span className="data-value">{driver.lap_time || '--:--'}</span>
                </div>
                <div className="data-item">
                  <span className="data-label">Banden</span>
                  <div className="tire-info">
                    <span 
                      className="tire-compound"
                      style={{ backgroundColor: getTireColor(driver.tire_compound) }}
                      title={driver.tire_compound}
                    >
                      {driver.tire_compound ? driver.tire_compound[0] : '?'}
                    </span>
                    <span className="tire-age">{driver.tire_age || 0}</span>
                  </div>
                </div>
              </div>
              <div className="driver-stats">
                <div className="stat-item">
                  <span className="stat-label">Pit</span>
                  <span className="stat-value">{driver.pit_stops || 0}</span>
                </div>
                <div className={`position-change ${driver.position_change > 0 ? 'gain' : driver.position_change < 0 ? 'loss' : 'neutral'}`}>
                  {driver.position_change > 0 && `+${driver.position_change}`}
                  {driver.position_change < 0 && `${driver.position_change}`}
                  {driver.position_change === 0 && '→'}
                </div>
              </div>
            </div>
          ))
        ) : (
          <div className="empty-state">
            <p>Wachten op race data...</p>
          </div>
        )}
      </div>
    </div>
  )
}
