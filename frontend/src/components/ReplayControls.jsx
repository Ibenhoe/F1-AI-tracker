import React from 'react';
import './ReplayControls.css';

/**
 * ReplayControls Component
 * Playback controls for race replay with speed adjustment,
 * frame seeking, and toggle options.
 */
const ReplayControls = ({
  isPlaying,
  playbackSpeed,
  currentFrame,
  totalFrames,
  showDRS,
  showTelemetry,
  focusMode,
  onPlayPause,
  onSpeedChange,
  onFrameChange,
  onDRSToggle,
  onTelemetryToggle,
  onFocusToggle,
}) => {
  const progressPercent = totalFrames > 0 ? (currentFrame / totalFrames) * 100 : 0;

  const handleSliderChange = (e) => {
    onFrameChange(Number(e.target.value));
  };

  const handleSpeedClick = (speed) => {
    onSpeedChange(speed);
  };

  return (
    <div className="replay-controls">
      {/* Progress Bar */}
      <div className="progress-section">
        <input
          type="range"
          min="0"
          max={totalFrames}
          value={Math.floor(currentFrame)}
          onChange={handleSliderChange}
          className="progress-slider"
          title="Click to seek"
        />
        <div
          className="progress-bar"
          style={{
            width: `${progressPercent}%`,
          }}
        />
      </div>

      {/* Controls Row */}
      <div className="controls-row">
        {/* Play/Pause Button */}
        <button
          className="control-btn play-pause-btn"
          onClick={onPlayPause}
          title={isPlaying ? 'Pause (SPACE)' : 'Play (SPACE)'}
        >
          {isPlaying ? (
            <span className="icon">⏸</span>
          ) : (
            <span className="icon">▶</span>
          )}
        </button>

        {/* Speed Controls */}
        <div className="speed-controls">
          {[0.1, 0.25, 0.5, 1, 2, 4].map((speed) => (
            <button
              key={speed}
              className={`speed-btn ${playbackSpeed === speed ? 'active' : ''}`}
              onClick={() => handleSpeedClick(speed)}
              title={`Speed: ${speed}x`}
            >
              {speed}x
            </button>
          ))}
        </div>

        {/* Spacer */}
        <div className="spacer"></div>

        {/* Frame Info */}
        <div className="frame-info">
          <span>
            {Math.floor(currentFrame)} / {totalFrames} frames
          </span>
          <span className="time-remaining">
            ~{formatFramesToTime(totalFrames - Math.floor(currentFrame))}
          </span>
        </div>

        {/* Spacer */}
        <div className="spacer"></div>

        {/* Toggle Buttons */}
        <button
          className={`control-btn toggle-btn ${showDRS ? 'active' : ''}`}
          onClick={onDRSToggle}
          title="Toggle DRS Zones (D)"
        >
          <span className="label">DRS</span>
        </button>

        <button
          className={`control-btn toggle-btn ${showTelemetry ? 'active' : ''}`}
          onClick={onTelemetryToggle}
          title="Toggle Telemetry (T)"
        >
          <span className="label">TEL</span>
        </button>

        <button
          className={`control-btn toggle-btn ${focusMode ? 'active' : ''}`}
          onClick={onFocusToggle}
          title="Toggle Focus Mode (F)"
        >
          <span className="label">FOCUS</span>
        </button>

        {/* Fullscreen Button (placeholder) */}
        <button
          className="control-btn"
          title="Fullscreen"
          onClick={() => {
            document.documentElement.requestFullscreen?.();
          }}
        >
          <span className="icon">⛶</span>
        </button>
      </div>
    </div>
  );
};

/**
 * Format remaining frames to approximate time
 * Assumes 120 FPS
 */
function formatFramesToTime(frames) {
  const seconds = Math.ceil(frames / 120);
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  if (minutes > 0) {
    return `${minutes}m ${remainingSeconds}s`;
  }
  return `${remainingSeconds}s`;
}

export default ReplayControls;
