import './WeatherWidget.css'

export default function WeatherWidget({ data }) {
  if (!data) {
    return <div className="weather-widget loading">Weer data laden...</div>
  }

  return (
    <div className="weather-widget">
      <div className="weather-header">
        <h2>⛅ Weer</h2>
      </div>
      <div className="weather-content">
        <div className="weather-main">
          <div className="weather-temp">
            <span className="temp-value">{data.temp}°</span>
            <span className="temp-unit">C</span>
          </div>
          <div className="weather-condition">{data.condition}</div>
        </div>
        <div className="weather-stats">
          <div className="stat">
            <span className="stat-label">Vochtigheid</span>
            <span className="stat-value">{data.humidity}%</span>
          </div>
          <div className="stat">
            <span className="stat-label">Windsnelheid</span>
            <span className="stat-value">{data.windSpeed} km/h</span>
          </div>
          <div className="stat">
            <span className="stat-label">Baantemp</span>
            <span className="stat-value">{data.trackTemp}°C</span>
          </div>
        </div>
      </div>
    </div>
  )
}
