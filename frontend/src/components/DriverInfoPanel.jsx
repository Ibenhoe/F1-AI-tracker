import React from 'react';
import './DriverInfoPanel.css';

/**
 * DriverInfoPanel Component
 * Displays detailed telemetry information for a selected driver
 */
const DriverInfoPanel = ({
  driver,
  driverData,
  frame,
  showTelemetry,
}) => {
  if (!driver || !driverData) {
    return <div className="driver-info-panel empty">No driver selected</div>;
  }

  return (
    <div className="driver-info-panel">
      {/* Driver Header */}
      <div className="info-header">
        <div className="driver-name">{driver}</div>
        <div className="driver-position">P{Math.floor(driverData.position)}</div>
      </div>

      {/* Telemetry Section */}
      {showTelemetry && (
        <div className="telemetry-section">
          {/* Speed */}
          <div className="telemetry-row">
            <span className="label">SPEED</span>
            <span className="value">{(driverData.speed || 0).toFixed(1)}</span>
            <span className="unit">km/h</span>
          </div>

          {/* Gear */}
          <div className="telemetry-row">
            <span className="label">GEAR</span>
            <span className="value gear">{driverData.gear || '-'}</span>
          </div>

          {/* Throttle & Brake */}
          <div className="telemetry-row">
            <span className="label">THROTTLE</span>
            <div className="progress-bar">
              <div
                className="progress-fill throttle"
                style={{
                  width: `${(driverData.throttle || 0) * 100}%`,
                }}
              />
            </div>
            <span className="value">{(driverData.throttle || 0).toFixed(0)}%</span>
          </div>

          <div className="telemetry-row">
            <span className="label">BRAKE</span>
            <div className="progress-bar">
              <div
                className="progress-fill brake"
                style={{
                  width: `${(driverData.brake || 0) * 100}%`,
                }}
              />
            </div>
            <span className="value">{(driverData.brake || 0).toFixed(0)}%</span>
          </div>

          {/* DRS */}
          <div className="telemetry-row">
            <span className="label">DRS</span>
            <span className={`drs-status ${driverData.drs ? 'active' : 'off'}`}>
              {driverData.drs ? '✓ ACTIVE' : '✗ OFF'}
            </span>
          </div>
        </div>
      )}

      {/* Tire Section */}
      <div className="tire-section">
        <div className="tire-header">TIRES</div>
        <div className="tire-info">
          <div className="tire-compound-display">
            <span className={`tire-badge ${getTireClass(driverData.tire_compound)}`}>
              {getTireShort(driverData.tire_compound)}
            </span>
            <span className="tire-name">{driverData.tire_compound || 'Unknown'}</span>
          </div>
          <div className="tire-age">
            <span className="label">Age:</span>
            <span className="value">{driverData.tire_age || 0} laps</span>
          </div>
        </div>
      </div>

      {/* Pit Stop Info */}
      {typeof driverData.pit_stops !== 'undefined' && (
        <div className="pit-stop-section">
          <span className="label">PIT STOPS</span>
          <span className="value">{driverData.pit_stops}</span>
        </div>
      )}

      {/* Gap Info */}
      {driverData.gap && (
        <div className="gap-section">
          <span className="label">GAP</span>
          <span className="value">{driverData.gap}</span>
        </div>
      )}

      {/* Status */}
      {driverData.status && driverData.status !== 'Running' && (
        <div className={`status-section status-${driverData.status.toLowerCase()}`}>
          <span>{driverData.status}</span>
        </div>
      )}
    </div>
  );
};

/**
 * Get CSS class for tire compound
 */
function getTireClass(compound) {
  if (!compound) return 'unknown';
  const lower = compound.toLowerCase();
  if (lower.includes('soft')) return 'soft';
  if (lower.includes('medium')) return 'medium';
  if (lower.includes('hard')) return 'hard';
  if (lower.includes('inter')) return 'inter';
  if (lower.includes('wet')) return 'wet';
  return 'unknown';
}

/**
 * Get short tire name
 */
function getTireShort(compound) {
  if (!compound) return '-';
  const lower = compound.toLowerCase();
  if (lower.includes('soft')) return 'S';
  if (lower.includes('medium')) return 'M';
  if (lower.includes('hard')) return 'H';
  if (lower.includes('inter')) return 'I';
  if (lower.includes('wet')) return 'W';
  return '-';
}

export default DriverInfoPanel;
