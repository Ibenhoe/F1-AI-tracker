import React, { useState, useEffect, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';
import TrackRenderer from '../components/TrackRenderer';
import ReplayControls from '../components/ReplayControls';
import ReplayLeaderboard from '../components/ReplayLeaderboard';
import DriverInfoPanel from '../components/DriverInfoPanel';
import Card from '../components/ui/Card';
import Badge from '../components/ui/Badge';
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
  const [playbackSpeed, setPlaybackSpeed] = useState(1);  // Default to 1x speed
  const [selectedDriver, setSelectedDriver] = useState(null);
  const [showDRS, setShowDRS] = useState(true);
  const [showTelemetry, setShowTelemetry] = useState(true);
  const [rightPanelTab, setRightPanelTab] = useState('standings');
  const [raceEvents, setRaceEvents] = useState([]);
  const [focusMode, setFocusMode] = useState(false);

  // Animation loop ref
  const animationRef = useRef(null);
  const lastFrameTimeRef = useRef(Date.now());
  const lastPitStopStateRef = useRef({});  // Use ref to avoid infinite loops

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

  // Track race events (pit stops, retirements, etc.)
  useEffect(() => {
    if (!currentFrame || !currentFrame.drivers) return;

    const newEvents = [];

    Object.entries(currentFrame.drivers).forEach(([code, driver]) => {
      const lastState = lastPitStopStateRef.current[code] || { pit_stops: 0, status: 'Running' };

      // Detect pit stop
      if (driver.pit_stops > lastState.pit_stops) {
        newEvents.push({
          id: `${currentFrame.lap}-${code}-pit`,
          type: 'pit_stop',
          message: `${code} pitted (Stop ${driver.pit_stops})`,
          lap: currentFrame.lap,
          driverCode: code,
          driverName: driver.driver_name || code,
          timestamp: new Date().toLocaleTimeString(),
        });
      }

      // Detect retirement
      if (driver.status === 'OUT' && lastState.status !== 'OUT') {
        newEvents.push({
          id: `${currentFrame.lap}-${code}-ret`,
          type: 'retirement',
          message: `${code} retired`,
          lap: currentFrame.lap,
          driverCode: code,
          driverName: driver.driver_name || code,
          timestamp: new Date().toLocaleTimeString(),
        });
      }

      // Update ref state (no setState, so no loop!)
      lastPitStopStateRef.current[code] = {
        pit_stops: driver.pit_stops,
        status: driver.status,
      };
    });

    if (newEvents.length > 0) {
      setRaceEvents((prev) => [...newEvents, ...prev].slice(0, 10)); // Keep last 10 events
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
        case 'f':
          setFocusMode(!focusMode);
          break;
        default:
          break;
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [isPlaying, raceData, showDRS, showTelemetry, focusMode]);

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
    <div className="space-y-6">
      {/* FOCUS MODE - FULL SCREEN TRACK ONLY */}
      {focusMode && (
        <div className="race-replay-focus-container">
          <div className="focus-track-fullscreen">
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
              focusMode={true}
              rotation={-90}
            />
          </div>

          {/* Floating controls bar at bottom */}
          <div className="focus-controls-container">
            <ReplayControls
              isPlaying={isPlaying}
              playbackSpeed={playbackSpeed}
              currentFrame={frameIndex}
              totalFrames={raceData?.frames?.length || 0}
              showDRS={showDRS}
              showTelemetry={showTelemetry}
              focusMode={focusMode}
              onPlayPause={() => setIsPlaying(!isPlaying)}
              onSpeedChange={setPlaybackSpeed}
              onFrameChange={setFrameIndex}
              onDRSToggle={() => setShowDRS(!showDRS)}
              onTelemetryToggle={() => setShowTelemetry(!showTelemetry)}
              onFocusToggle={() => setFocusMode(!focusMode)}
            />
          </div>
        </div>
      )}

      {/* NORMAL VIEW - Layout with track and standings */}
      {!focusMode && (
        <>
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div className="space-y-1">
          <h1 className="text-xl font-semibold tracking-tight">{raceData?.raceName || 'Race Replay'}</h1>
          <p className="text-sm text-neutral-500 dark:text-neutral-400">
            Interactive race replay with real-time positions and telemetry analysis.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Badge variant="neutral">LAP {currentLap}</Badge>
          {currentFrame && (
            <Badge variant="neutral">
              {formatTime(currentFrame.raceTime)}
            </Badge>
          )}
          <Badge variant={isPlaying ? "warning" : "neutral"}>
            {isPlaying ? "Playing" : "Paused"}
          </Badge>
          <Badge variant="neutral">{playbackSpeed.toFixed(2)}x</Badge>
        </div>
      </div>

      {/* MAIN GRID */}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-12">
        {/* TRACK VISUALIZATION - Main */}
        <Card className="xl:col-span-8 p-5 aspect-video" clip>
          <div className="w-full h-full flex flex-col">
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
              focusMode={focusMode}
              rotation={-90}
            />
          </div>
        </Card>

        {/* RIGHT PANEL - Standings & Info */}
        <Card className="xl:col-span-4 p-5" clip>
          <div className="flex h-full min-w-0 flex-col gap-4">
            {/* Tabs */}
            <div className="flex flex-wrap gap-2">
              {[
                { id: 'standings', label: 'Standings' },
                { id: 'driver', label: 'Driver Info' },
                { id: 'events', label: 'Events' }
              ].map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setRightPanelTab(tab.id)}
                  className={[
                    "rounded-full border px-3 py-1 text-sm font-medium transition",
                    rightPanelTab === tab.id
                      ? "border-neutral-900 bg-neutral-900 text-white dark:border-neutral-200 dark:bg-neutral-100 dark:text-neutral-900"
                      : "border-neutral-200 bg-white text-neutral-700 hover:bg-neutral-50 dark:border-neutral-800 dark:bg-neutral-950/40 dark:text-neutral-200 dark:hover:bg-neutral-900/40",
                  ].join(" ")}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            {/* Tab Content */}
            <div className="min-h-0 min-w-0 flex-1 overflow-auto">
              {rightPanelTab === 'standings' && (
                <div>
                  <ReplayLeaderboard
                    drivers={currentFrame?.drivers || []}
                    selectedDriver={selectedDriver}
                    onDriverSelect={setSelectedDriver}
                  />
                </div>
              )}

              {rightPanelTab === 'driver' && selectedDriver && currentFrame && (
                <div>
                  <DriverInfoPanel
                    driver={selectedDriver}
                    driverData={currentFrame.drivers[selectedDriver]}
                    frame={currentFrame}
                    showTelemetry={showTelemetry}
                  />
                </div>
              )}

              {rightPanelTab === 'driver' && !selectedDriver && (
                <div className="flex items-center justify-center h-full text-neutral-500">
                  <p className="text-sm">Select a driver to view details</p>
                </div>
              )}

              {rightPanelTab === 'events' && (
                <div className="space-y-2">
                  {raceEvents.length === 0 ? (
                    <div className="flex items-center justify-center h-full text-neutral-500">
                      <p className="text-sm">No events yet</p>
                    </div>
                  ) : (
                    raceEvents.map((event) => (
                      <div
                        key={event.id}
                        className="p-2 rounded bg-neutral-100 dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800"
                      >
                        <div className="flex items-start justify-between gap-2">
                          <div className="flex-1 min-w-0">
                            <p className="text-xs font-medium text-neutral-900 dark:text-neutral-100">
                              {event.type === 'pit_stop' && '🛠 '}
                              {event.type === 'retirement' && '❌ '}
                              {event.message}
                            </p>
                            <p className="text-xs text-neutral-500 dark:text-neutral-400">
                              Lap {event.lap}
                            </p>
                          </div>
                          <span className="flex-shrink-0 text-xs text-neutral-400">
                            {event.timestamp}
                          </span>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              )}
            </div>
          </div>
        </Card>
      </div>

      {/* CONTROLS */}
      <Card className="p-5" clip>
        <ReplayControls
          isPlaying={isPlaying}
          playbackSpeed={playbackSpeed}
          currentFrame={frameIndex}
          totalFrames={raceData?.frames?.length || 0}
          showDRS={showDRS}
          showTelemetry={showTelemetry}
          focusMode={focusMode}
          onPlayPause={() => setIsPlaying(!isPlaying)}
          onSpeedChange={setPlaybackSpeed}
          onFrameChange={setFrameIndex}
          onDRSToggle={() => setShowDRS(!showDRS)}
          onTelemetryToggle={() => setShowTelemetry(!showTelemetry)}
          onFocusToggle={() => setFocusMode(!focusMode)}
        />
      </Card>

      {/* KEYBOARD SHORTCUTS */}
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 md:grid-cols-6 text-xs text-neutral-500">
        <div><kbd className="px-2 py-1 bg-neutral-200 dark:bg-neutral-800 rounded">SPACE</kbd> Play/Pause</div>
        <div><kbd className="px-2 py-1 bg-neutral-200 dark:bg-neutral-800 rounded">←/→</kbd> Seek</div>
        <div><kbd className="px-2 py-1 bg-neutral-200 dark:bg-neutral-800 rounded">+/-</kbd> Speed</div>
        <div><kbd className="px-2 py-1 bg-neutral-200 dark:bg-neutral-800 rounded">D</kbd> DRS</div>
        <div><kbd className="px-2 py-1 bg-neutral-200 dark:bg-neutral-800 rounded">T</kbd> Telemetry</div>
        <div><kbd className="px-2 py-1 bg-neutral-200 dark:bg-neutral-800 rounded">F</kbd> Focus</div>
        <div><kbd className="px-2 py-1 bg-neutral-200 dark:bg-neutral-800 rounded">R</kbd> Reset</div>
      </div>
        </>
      )}
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
