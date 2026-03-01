import React, { useState, useEffect, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';
import TrackRenderer from '../components/TrackRenderer';
import ReplayControls from '../components/ReplayControls';
import ReplayLeaderboard from '../components/ReplayLeaderboard';
import DriverInfoPanel from '../components/DriverInfoPanel';
import Card from '../components/ui/Card';
import Badge from '../components/ui/Badge';
import { getTeamColor } from '../utils/teamColors';
import './RaceReplay.css';

// ─── Win-probability model (position + gap + tire age) ───────────────────────
const POSITION_WEIGHTS = [100, 38, 18, 10, 6, 4, 2.5, 1.8, 1.2, 0.9,
                          0.7, 0.55, 0.45, 0.35, 0.28, 0.22, 0.16, 0.12, 0.08, 0.05];

function computePredictions(currentFrame) {
  if (!currentFrame?.drivers) return [];

  const entries = Object.entries(currentFrame.drivers)
    .filter(([, d]) => d.status !== 'OUT');
  if (entries.length === 0) return [];

  const weighted = entries.map(([code, d]) => {
    const pos   = Math.max(1, Math.round(d.position || 20));
    const base  = POSITION_WEIGHTS[Math.min(pos - 1, POSITION_WEIGHTS.length - 1)];
    const gap   = parseFloat((d.gap || '+0').replace('+', '')) || 0;
    const gapF  = gap < 5 ? 1.0 : gap < 20 ? 0.88 : gap < 45 ? 0.65 : 0.35;
    const tireF = (d.tire_age || 0) < 10 ? 1.05 :
                  (d.tire_age || 0) < 25 ? 1.0  :
                  (d.tire_age || 0) < 40 ? 0.92 : 0.82;
    return { code, d, weight: base * gapF * tireF };
  });

  const total = weighted.reduce((s, w) => s + w.weight, 0) || 1;
  return weighted
    .map(({ code, d, weight }) => ({
      code,
      name:     d.driver_name || code,
      team:     d.team || '',
      position: Math.round(d.position || 20),
      probability: Math.round((weight / total) * 100),
      tireCompound: d.tire_compound || 'MEDIUM',
      tireAge:  d.tire_age || 0,
      gap:      d.gap || '+0.000',
    }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, 10);
}

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
  const [realTimeMode, setRealTimeMode] = useState(false);
  // ML model predictions keyed by lap number (fetched from backend)
  const [lapPredictions, setLapPredictions] = useState(null);   // null = loading, {} = loaded
  const [predictionsLoading, setPredictionsLoading] = useState(false);

  // Animation refs kept for compatibility but loop is now owned by TrackRenderer
  const animationRef = useRef(null);
  const lastFrameTimeRef = useRef(performance.now());
  const lastPitStopStateRef = useRef({});  // Use ref to avoid infinite loops
  const smoothedGapsRef = useRef({});  // Track smoothed gaps per driver
  const lastProcessedFrameRef = useRef(0);  // Detect large seeks to reset events
  // Real-time sync: records the wall-clock and raceTime when sync started
  const syncStartRef = useRef(null); // { wallMs, raceTime }

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

  // Fetch ML model predictions per lap from backend (runs in background)
  useEffect(() => {
    const fetchMLPredictions = async () => {
      try {
        setPredictionsLoading(true);
        const raceNum = searchParams.get('race') || 21;
        const res = await fetch(`/api/race/lap-predictions?race=${raceNum}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (data.predictions_by_lap) {
          setLapPredictions(data.predictions_by_lap);
        }
      } catch (err) {
        console.warn('[RaceReplay] ML predictions unavailable, using heuristic fallback:', err.message);
        setLapPredictions({});  // empty → triggers fallback
      } finally {
        setPredictionsLoading(false);
      }
    };
    fetchMLPredictions();
  }, [searchParams]);

  // Animation loop is now fully owned by TrackRenderer (single RAF, no React state updates per frame)
  // frameIndex state is only updated by TrackRenderer's onFrameChange callback (for slider/lap display)

  // Frame index for slider / lap display (updated by TrackRenderer callback, not per-frame setState)
  const frameIdx = raceData ? Math.min(Math.floor(frameIndex), raceData.frames.length - 1) : 0;
  const currentFrame = raceData?.frames[frameIdx] ?? null;
  const totalLaps = raceData?.frames?.[raceData.frames.length - 1]?.lap ?? null;

  // Called by TrackRenderer to sync slider + lap counter (~every 10 frames)
  const handleFrameChange = (idx) => {
    setFrameIndex(idx);
    if (idx >= (raceData?.frames.length ?? 1) - 1) {
      setIsPlaying(false);
    }
  };

  // Update current lap from frame
  useEffect(() => {
    if (currentFrame && currentFrame.lap) {
      setCurrentLap(currentFrame.lap);
    }
  }, [currentFrame]);

  // Track race events (pit stops, retirements, etc.)
  useEffect(() => {
    if (!currentFrame || !currentFrame.drivers) return;

    // If user scrubbed significantly, reset event history to avoid duplicates
    const jumpSize = Math.abs(frameIdx - lastProcessedFrameRef.current);
    if (jumpSize > 5) {
      setRaceEvents([]);
      lastPitStopStateRef.current = {};
    }
    lastProcessedFrameRef.current = frameIdx;

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

      // Calculate smoothed gap (for visual smoothing on pitstops)
      const currentGapStr = driver.gap || '+0.000';
      const currentGap = parseFloat(currentGapStr.replace('+', '')) || 0;
      const previousSmoothedGap = smoothedGapsRef.current[code] || currentGap;
      
      // If gap jump > 2 seconds, smooth it gradually
      let smoothedGap = currentGap;
      if (Math.abs(currentGap - previousSmoothedGap) > 2.0) {
        // Interpolate 10% toward new target
        smoothedGap = previousSmoothedGap + (currentGap - previousSmoothedGap) * 0.1;
      } else {
        smoothedGap = currentGap;
      }
      
      smoothedGapsRef.current[code] = smoothedGap;

      // Update ref state
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
        case 's': {
          const nextRT = !realTimeMode;
          if (nextRT && currentFrame) {
            // Arm sync: record current wall time and the current frame's raceTime
            syncStartRef.current = {
              wallMs:    performance.now(),
              raceTime:  currentFrame.raceTime || 0,
            };
            setIsPlaying(false); // TrackRenderer stops its own RAF; we drive frames
          } else {
            syncStartRef.current = null;
            // Resume normal playback from wherever we are
            setIsPlaying(true);
          }
          setRealTimeMode(nextRT);
          break;
        }
        default:
          break;
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [isPlaying, raceData, showDRS, showTelemetry, focusMode, realTimeMode, currentFrame]);

  // ── Real-time sync: advance frameIndex by elapsed wall-clock seconds ─────────
  useEffect(() => {
    if (!realTimeMode || !raceData) return;

    const FRAMES_PER_LAP = 120;
    const SECONDS_PER_LAP = 90; // approximate value used in _build_replay_frames

    const interval = setInterval(() => {
      if (!syncStartRef.current) return;
      const elapsedSec = (performance.now() - syncStartRef.current.wallMs) / 1000;
      const targetRaceTime = syncStartRef.current.raceTime + elapsedSec;
      // frameIdx = raceTime * (FRAMES_PER_LAP / SECONDS_PER_LAP)
      const targetIdx = Math.round(targetRaceTime * FRAMES_PER_LAP / SECONDS_PER_LAP);
      const clampedIdx = Math.min(Math.max(0, targetIdx), raceData.frames.length - 1);
      setFrameIndex(clampedIdx);

      if (clampedIdx >= raceData.frames.length - 1) {
        // Race finished
        setRealTimeMode(false);
        syncStartRef.current = null;
      }
    }, 250); // update 4× per second — smooth enough for race sync

    return () => clearInterval(interval);
  }, [realTimeMode, raceData]);

  if (loading) {
    return (
      <div className="space-y-6">
        <Card className="p-8" clip>
          <div className="flex items-center gap-4">
            <div className="h-10 w-10 rounded-full border-4 border-neutral-200 border-t-red-500 animate-spin dark:border-neutral-800 dark:border-t-red-400" />
            <div>
              <div className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">Loading race replay</div>
              <div className="mt-1 text-sm text-neutral-500 dark:text-neutral-400">Fetching frame data from the server…</div>
            </div>
          </div>
        </Card>
      </div>
    );
  }

  if (error || !raceData) {
    return (
      <div className="space-y-6">
        <Card className="p-6" clip>
          <div className="flex items-start justify-between gap-3">
            <div>
              <div className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">
                {error ? 'Error loading replay' : 'No data available'}
              </div>
              <div className="mt-1 text-sm text-neutral-500 dark:text-neutral-400">
                {error || 'Could not load race data for replay.'}
              </div>
            </div>
            <Badge variant="danger">Error</Badge>
          </div>
          <button
            type="button"
            onClick={() => window.location.reload()}
            className="mt-4 rounded-full border border-neutral-200 px-4 py-1.5 text-sm font-medium text-neutral-700 hover:bg-neutral-50 dark:border-neutral-700 dark:text-neutral-300 dark:hover:bg-neutral-900"
          >
            Retry
          </button>
        </Card>
      </div>
    );
  }

  // ── Resolve which predictions to show ─────────────────────────────────────
  // Prefer real ML model data (fetched from /api/race/lap-predictions).
  // Fall back to the client-side heuristic while predictions are loading or
  // if the backend returned no data for the current lap.
  const displayPredictions = (() => {
    const mlPreds = lapPredictions?.[String(currentLap)];
    if (mlPreds && mlPreds.length > 0) {
      return mlPreds.slice(0, 10).map(p => ({
        code:          p.driver_code,
        name:          p.driver_name,
        team:          p.team,
        probability:   Math.round(p.confidence),
        position:      p.predicted_position,
        tireCompound:  'MEDIUM',
        tireAge:       0,
        gap:           '',
      }));
    }
    return computePredictions(currentFrame);
  })();

  return (
    <div className="space-y-6">
      {/* FOCUS MODE - FULL SCREEN TRACK + FLOATING LEADERBOARD LEFT */}
      {focusMode && (
        <div className="race-replay-focus-container">
          <div className="focus-track-fullscreen">
            <TrackRenderer
              ref={canvasRef}
              frames={raceData.frames}
              frameIndex={frameIndex}
              isPlaying={realTimeMode ? false : isPlaying}
              playbackSpeed={playbackSpeed}
              onFrameChange={handleFrameChange}
              currentFrame={currentFrame}
              trackData={raceData.trackData}
              drsZones={raceData.drsZones}
              showDRS={showDRS}
              selectedDriver={selectedDriver}
              onDriverSelect={setSelectedDriver}
              focusMode={true}
              rotation={-90}
              smoothedGaps={smoothedGapsRef.current}
            />
          </div>

          {/* Floating leaderboard – left side */}
          <div
            className="absolute top-4 left-4 z-20 w-52 rounded-xl border border-neutral-700/60 bg-neutral-950/80 backdrop-blur-md shadow-xl overflow-hidden"
            style={{ maxHeight: 'calc(100% - 96px)' }}
          >
            <div className="overflow-y-auto" style={{ maxHeight: 'inherit' }}>
              <ReplayLeaderboard
                drivers={currentFrame?.drivers || []}
                selectedDriver={selectedDriver}
                onDriverSelect={setSelectedDriver}
                currentLap={currentLap}
                totalLaps={totalLaps}
              />
            </div>
          </div>

          {/* Floating AI predictions panel – right side */}
          <div
            className="absolute top-4 right-4 z-20 w-56 rounded-xl border border-neutral-700/60 bg-neutral-950/80 backdrop-blur-md shadow-xl overflow-hidden"
            style={{ maxHeight: 'calc(100% - 96px)' }}
          >
            <div className="px-3 py-2.5 border-b border-neutral-700/50">
              <div className="flex items-center justify-between gap-1">
                <div className="text-[10px] font-semibold uppercase tracking-widest text-neutral-400">AI Win Prediction</div>
                {predictionsLoading && (
                  <div className="h-2.5 w-2.5 rounded-full border-2 border-neutral-600 border-t-blue-400 animate-spin" />
                )}
              </div>
              <div className="text-[9px] text-neutral-500 mt-0.5">
                {lapPredictions && !predictionsLoading ? 'Real ML model' : 'Heuristic'} · Lap {currentLap} / {totalLaps || '–'}
              </div>
            </div>
            <div className="overflow-y-auto px-2 py-2 space-y-1.5" style={{ maxHeight: 'calc(100% - 48px)' }}>
              {displayPredictions.map((p, idx) => {
                const teamColor = getTeamColor(p.team) || '#6b7280';
                return (
                  <div key={p.code} className="flex items-center gap-2">
                    <span className="w-4 shrink-0 text-[10px] tabular-nums text-neutral-500 text-right">{idx + 1}</span>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between gap-1 mb-0.5">
                        <span className="text-[11px] font-semibold text-neutral-100 truncate">{p.code}</span>
                        <span className="text-[11px] font-bold tabular-nums" style={{ color: teamColor }}>{p.probability}%</span>
                      </div>
                      <div className="h-1 w-full rounded-full bg-neutral-800 overflow-hidden">
                        <div
                          className="h-full rounded-full transition-[width] duration-300"
                          style={{ width: `${p.probability}%`, background: teamColor }}
                        />
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Real-time sync badge */}
          {realTimeMode && (
            <div className="absolute top-4 left-1/2 -translate-x-1/2 z-30 flex items-center gap-1.5 rounded-full border border-red-500/40 bg-red-500/15 px-3 py-1">
              <span className="inline-block h-1.5 w-1.5 rounded-full bg-red-500 animate-pulse" />
              <span className="text-[11px] font-semibold text-red-400 tracking-wide">LIVE SYNC</span>
            </div>
          )}

          {/* Floating controls bar at bottom */}
          <div className="focus-controls-container">
            <ReplayControls
              isPlaying={realTimeMode ? false : isPlaying}
              playbackSpeed={playbackSpeed}
              currentFrame={frameIndex}
              totalFrames={raceData?.frames?.length || 0}
              showDRS={showDRS}
              showTelemetry={showTelemetry}
              focusMode={focusMode}
              realTimeMode={realTimeMode}
              onPlayPause={() => {
                if (realTimeMode) return; // RT mode controls playback itself
                setIsPlaying(!isPlaying);
              }}
              onSpeedChange={setPlaybackSpeed}
              onFrameChange={(idx) => {
                setFrameIndex(idx);
                // If user scrubs while in RT mode, re-arm sync at new position
                if (realTimeMode && raceData?.frames?.[idx]) {
                  syncStartRef.current = {
                    wallMs:   performance.now(),
                    raceTime: raceData.frames[idx].raceTime || 0,
                  };
                }
              }}
              onDRSToggle={() => setShowDRS(!showDRS)}
              onTelemetryToggle={() => setShowTelemetry(!showTelemetry)}
              onFocusToggle={() => setFocusMode(!focusMode)}
              onRealTimeToggle={() => {
                const nextRT = !realTimeMode;
                if (nextRT && currentFrame) {
                  syncStartRef.current = { wallMs: performance.now(), raceTime: currentFrame.raceTime || 0 };
                  setIsPlaying(false);
                } else {
                  syncStartRef.current = null;
                  setIsPlaying(true);
                }
                setRealTimeMode(nextRT);
              }}
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
              isPlaying={realTimeMode ? false : isPlaying}
              playbackSpeed={playbackSpeed}
              onFrameChange={handleFrameChange}
              currentFrame={currentFrame}
              trackData={raceData.trackData}
              drsZones={raceData.drsZones}
              showDRS={showDRS}
              selectedDriver={selectedDriver}
              onDriverSelect={setSelectedDriver}
              focusMode={focusMode}
              rotation={-90}
              smoothedGaps={smoothedGapsRef.current}
            />
          </div>
        </Card>

        {/* RIGHT PANEL – Standings, Driver Info & Events */}
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
                <ReplayLeaderboard
                  drivers={currentFrame?.drivers || []}
                  selectedDriver={selectedDriver}
                  onDriverSelect={setSelectedDriver}
                  currentLap={currentLap}
                  totalLaps={totalLaps}
                />
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
                    <div className="flex items-center justify-center py-10 text-neutral-500">
                      <p className="text-sm">No events yet</p>
                    </div>
                  ) : (
                    raceEvents.map((event) => (
                      <div
                        key={event.id}
                        className={[
                          'flex items-start gap-2.5 rounded-xl border p-3',
                          event.type === 'pit_stop'
                            ? 'border-amber-200/60 bg-amber-50/60 dark:border-amber-500/20 dark:bg-amber-500/5'
                            : 'border-red-200/60 bg-red-50/60 dark:border-red-500/20 dark:bg-red-500/5',
                        ].join(' ')}
                      >
                        <span className="mt-0.5 flex-shrink-0 text-base leading-none">
                          {event.type === 'pit_stop' ? '🔧' : '🚩'}
                        </span>
                        <div className="flex-1 min-w-0">
                          <p className="text-xs font-semibold text-neutral-900 dark:text-neutral-100">
                            {event.message}
                          </p>
                          <p className="mt-0.5 text-[11px] text-neutral-500 dark:text-neutral-400">
                            Lap {event.lap}
                          </p>
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
          isPlaying={realTimeMode ? false : isPlaying}
          playbackSpeed={playbackSpeed}
          currentFrame={frameIndex}
          totalFrames={raceData?.frames?.length || 0}
          showDRS={showDRS}
          showTelemetry={showTelemetry}
          focusMode={focusMode}
          realTimeMode={realTimeMode}
          onPlayPause={() => {
            if (realTimeMode) return;
            setIsPlaying(!isPlaying);
          }}
          onSpeedChange={setPlaybackSpeed}
          onFrameChange={(idx) => {
            setFrameIndex(idx);
            if (realTimeMode && raceData?.frames?.[idx]) {
              syncStartRef.current = {
                wallMs:   performance.now(),
                raceTime: raceData.frames[idx].raceTime || 0,
              };
            }
          }}
          onDRSToggle={() => setShowDRS(!showDRS)}
          onTelemetryToggle={() => setShowTelemetry(!showTelemetry)}
          onFocusToggle={() => setFocusMode(!focusMode)}
          onRealTimeToggle={() => {
            const nextRT = !realTimeMode;
            if (nextRT && currentFrame) {
              syncStartRef.current = { wallMs: performance.now(), raceTime: currentFrame.raceTime || 0 };
              setIsPlaying(false);
            } else {
              syncStartRef.current = null;
              setIsPlaying(true);
            }
            setRealTimeMode(nextRT);
          }}
        />
      </Card>

      {/* KEYBOARD SHORTCUTS */}
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 md:grid-cols-7 text-xs text-neutral-500">
        <div><kbd className="px-2 py-1 bg-neutral-200 dark:bg-neutral-800 rounded">SPACE</kbd> Play/Pause</div>
        <div><kbd className="px-2 py-1 bg-neutral-200 dark:bg-neutral-800 rounded">←/→</kbd> Seek</div>
        <div><kbd className="px-2 py-1 bg-neutral-200 dark:bg-neutral-800 rounded">+/-</kbd> Speed</div>
        <div><kbd className="px-2 py-1 bg-neutral-200 dark:bg-neutral-800 rounded">D</kbd> DRS</div>
        <div><kbd className="px-2 py-1 bg-neutral-200 dark:bg-neutral-800 rounded">T</kbd> Telemetry</div>
        <div><kbd className="px-2 py-1 bg-neutral-200 dark:bg-neutral-800 rounded">F</kbd> Focus + Predictions</div>
        <div><kbd className="px-2 py-1 bg-neutral-200 dark:bg-neutral-800 rounded">S</kbd> Sync to real race</div>
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
