import React, { useState, useEffect, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';
import TrackRenderer from '../components/TrackRenderer';
import ReplayControls from '../components/ReplayControls';
import ReplayLeaderboard from '../components/ReplayLeaderboard';
import DriverInfoPanel from '../components/DriverInfoPanel';
import './RaceReplay.css';

/**
 * RaceReplay Component
 * Displays an interactive replay of a F1 race with track visualization,
 * real-time driver positions, and detailed telemetry data.
 */
const RaceReplay = () => {
  const [searchParams] = useSearchParams();
  const canvasRef = useRef(null);
  
  // State management
  const [raceData, setRaceData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [currentLap, setCurrentLap] = useState(0);
  const [frameIndex, setFrameIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(0.1);  // Default to 1/10 speed (very slow) - race will take ~12 minutes
  const [selectedDriver, setSelectedDriver] = useState(null);
  const [showDRS, setShowDRS] = useState(true);
  const [showTelemetry, setShowTelemetry] = useState(true);

  // Animation loop ref
  const animationRef = useRef(null);
  const lastFrameTimeRef = useRef(Date.now());

  // Load race data from API
  useEffect(() => {
    const loadRaceData = async () => {
      try {
        setLoading(true);
        const raceNum = searchParams.get('race') || 21;
        
        const response = await fetch(`/api/race/replay-data?race=${raceNum}`);
        if (!response.ok) {
          throw new Error(`Failed to load race data: ${response.status}`);
        }
        
        const data = await response.json();
        setRaceData(data);
        setCurrentLap(1);
        setFrameIndex(0);
        setError(null);
      } catch (err) {
        setError(err.message);
        console.error('Error loading race data:', err);
      } finally {
        setLoading(false);
      }
    };

    loadRaceData();
  }, [searchParams]);

  // Animation loop for playback
  useEffect(() => {
    if (!isPlaying || !raceData || raceData.frames.length === 0) {
      return;
    }

    const animate = () => {
      const now = Date.now();
      const deltaTime = (now - lastFrameTimeRef.current) / 1000; // Convert to seconds
      lastFrameTimeRef.current = now;

      // Calculate frame advance based on playback speed
      // Frames are at 120fps (ultra-smooth cinema-quality animation)
      // Multiply by playbackSpeed to control overall race speed (0.5 = half speed = slower)
      const framesPerSecond = 120;
      const frameAdvance = deltaTime * framesPerSecond * playbackSpeed;
      
      setFrameIndex((prevIndex) => {
        const newIndex = prevIndex + frameAdvance;
        
        if (newIndex >= raceData.frames.length) {
          // End of replay
          setIsPlaying(false);
          return raceData.frames.length - 1;
        }
        
        return newIndex;
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    lastFrameTimeRef.current = Date.now();
    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isPlaying, playbackSpeed, raceData]);

  // Get current frame data
  const getCurrentFrame = () => {
    if (!raceData || raceData.frames.length === 0) return null;
    const index = Math.floor(frameIndex);
    return raceData.frames[Math.min(index, raceData.frames.length - 1)];
  };

  const currentFrame = getCurrentFrame();

  // Update current lap from frame
  useEffect(() => {
    if (currentFrame && currentFrame.lap) {
      setCurrentLap(currentFrame.lap);
    }
  }, [currentFrame]);

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e) => {
      switch (e.key) {
        case ' ':
          e.preventDefault();
          setIsPlaying(!isPlaying);
          break;
        case 'ArrowLeft':
          setFrameIndex((prev) => Math.max(0, prev - 30));
          break;
        case 'ArrowRight':
          if (raceData) {
            setFrameIndex((prev) =>
              Math.min(prev + 30, raceData.frames.length - 1)
            );
          }
          break;
        case '+':
          setPlaybackSpeed((prev) => Math.min(prev + 0.25, 4));
          break;
        case '-':
          setPlaybackSpeed((prev) => Math.max(prev - 0.25, 0.25));
          break;
        case 'd':
          setShowDRS(!showDRS);
          break;
        case 't':
          setShowTelemetry(!showTelemetry);
          break;
        case 'r':
          setFrameIndex(0);
          setIsPlaying(false);
          break;
        default:
          break;
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [isPlaying, raceData, showDRS, showTelemetry]);

  if (loading) {
    return (
      <div className="race-replay-container">
        <div className="loading">
          <div className="spinner"></div>
          <p>Loading race replay data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="race-replay-container">
        <div className="error">
          <h2>Error Loading Replay</h2>
          <p>{error}</p>
          <button onClick={() => window.location.reload()}>Retry</button>
        </div>
      </div>
    );
  }

  if (!raceData) {
    return (
      <div className="race-replay-container">
        <div className="error">
          <h2>No Data Available</h2>
          <p>Could not load race data for replay.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="race-replay-container">
      {/* Main Track Visualization */}
      <div className="replay-main">
        <div className="track-canvas-wrapper">
          <TrackRenderer
            ref={canvasRef}
            frames={raceData.frames}
            frameIndex={frameIndex}
            currentFrame={currentFrame}
            trackData={raceData.trackData}
            drsZones={raceData.drsZones}
            showDRS={showDRS}
            selectedDriver={selectedDriver}
            onDriverSelect={setSelectedDriver}
          />
          
          {/* Race Info Overlay */}
          <div className="race-info-overlay">
            <div className="race-header">
              <h1>{raceData.raceName}</h1>
              <p className="race-year">{raceData.year}</p>
            </div>

            <div className="lap-time-display">
              <div className="lap-info">
                <span className="label">LAP</span>
                <span className="value">{currentLap}</span>
              </div>
              {currentFrame && (
                <div className="time-info">
                  <span className="label">TIME</span>
                  <span className="value">
                    {formatTime(currentFrame.raceTime)}
                  </span>
                </div>
              )}
            </div>

            {/* Track Status */}
            {currentFrame && currentFrame.trackStatus && (
              <div className={`track-status status-${currentFrame.trackStatus.toLowerCase()}`}>
                {currentFrame.trackStatus}
              </div>
            )}
          </div>
        </div>

        {/* Right Side Panel */}
        <div className="replay-right-panel">
          {/* Leaderboard */}
          <div className="leaderboard-section">
            <h3>STANDINGS</h3>
            <ReplayLeaderboard
              drivers={currentFrame?.drivers || []}
              selectedDriver={selectedDriver}
              onDriverSelect={setSelectedDriver}
            />
          </div>

          {/* Selected Driver Info */}
          {selectedDriver && currentFrame && (
            <div className="driver-info-section">
              <h3>DRIVER INFO</h3>
              <DriverInfoPanel
                driver={selectedDriver}
                driverData={currentFrame.drivers[selectedDriver]}
                frame={currentFrame}
                showTelemetry={showTelemetry}
              />
            </div>
          )}
        </div>
      </div>

      {/* Bottom Controls */}
      <div className="replay-bottom">
        <ReplayControls
          isPlaying={isPlaying}
          playbackSpeed={playbackSpeed}
          currentFrame={frameIndex}
          totalFrames={raceData.frames.length}
          showDRS={showDRS}
          showTelemetry={showTelemetry}
          onPlayPause={() => setIsPlaying(!isPlaying)}
          onSpeedChange={setPlaybackSpeed}
          onFrameChange={setFrameIndex}
          onDRSToggle={() => setShowDRS(!showDRS)}
          onTelemetryToggle={() => setShowTelemetry(!showTelemetry)}
        />
      </div>

      {/* Keyboard Shortcuts Legend */}
      <div className="keyboard-legend">
        <div className="legend-item">SPACE: Play/Pause</div>
        <div className="legend-item">←/→: Rewind/Forward</div>
        <div className="legend-item">+/-: Speed</div>
        <div className="legend-item">D: DRS</div>
        <div className="legend-item">T: Telemetry</div>
        <div className="legend-item">R: Reset</div>
      </div>
    </div>
  );
};

/**
 * Format time in seconds to HH:MM:SS
 */
function formatTime(seconds) {
  if (!seconds && seconds !== 0) return '00:00:00';
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

export default RaceReplay;
