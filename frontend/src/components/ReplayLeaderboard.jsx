import React from 'react';
import './ReplayLeaderboard.css';

/**
 * ReplayLeaderboard Component
 * Displays current race standings with driver info and gap indicators
 */
const ReplayLeaderboard = ({
  drivers,
  selectedDriver,
  onDriverSelect,
  focusMode = false,
}) => {
  // Sort drivers by position
  const sortedDrivers = Object.entries(drivers || {})
    .map(([code, driver]) => ({ code, ...driver }))
    .sort((a, b) => a.position - b.position);

  return (
    <div className={`replay-leaderboard ${focusMode ? 'focus-mode' : ''}`}>
      {sortedDrivers.length === 0 ? (
        <div className="empty-state">No driver data available</div>
      ) : (
        <div className={`leaderboard-list ${focusMode ? 'focus-list' : ''}`}>
          {sortedDrivers.map((driver) => (
            <div
              key={driver.code}
              className={`leaderboard-entry ${
                selectedDriver === driver.code ? 'selected' : ''
              } ${driver.status === 'OUT' ? 'retired' : ''}`}
              onClick={() =>
                onDriverSelect(
                  selectedDriver === driver.code ? null : driver.code
                )
              }
            >
              <div className="position-badge">{Math.floor(driver.position)}</div>
              
              <div className="driver-info">
                <div className="driver-code">{driver.code}</div>
                <div className="driver-name">{driver.driver_name || driver.code}</div>
                <div className="driver-team">{driver.team || 'Unknown'}</div>
              </div>

              <div className="driver-stats">
                {driver.gap && (
                  <div className={`gap ${driver.position === 1 ? 'leader' : ''}`}>
                    <span className="gap-label">GAP</span>
                    <span className="gap-value">{driver.gap}</span>
                  </div>
                )}
              </div>

              <div className="tire-info">
                <span
                  className={`tire-badge ${getTireClass(
                    driver.tire_compound
                  )}`}
                >
                  {getTireShort(driver.tire_compound)}
                </span>
                {driver.tire_age && driver.tire_age > 0 && (
                  <span className="tire-age" title={`Lap ${driver.tire_age} on tire`}>
                    L{driver.tire_age}
                  </span>
                )}
              </div>

              {driver.pit_stops && driver.pit_stops > 0 && (
                <div className="pit-stops-badge" title={`Pit stops: ${driver.pit_stops}`}>
                  🛠 {driver.pit_stops}
                </div>
              )}

              {driver.status === 'OUT' && (
                <div className="status-badge">DNF</div>
              )}

              {driver.drs && (
                <div className="drs-indicator">DRS</div>
              )}
            </div>
          ))}
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

export default ReplayLeaderboard;
