// src/pages/RaceReplay.jsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";

import Card from "../components/ui/Card";
import Badge from "../components/ui/Badge";

import {
  useRaceReplayData,
  useLapPredictions,
  useRaceEvents,
  useKeyboardShortcuts,
  useRealTimeSync,
  computePredictions,
  FocusOverlay,
  NormalView,
} from "../components/racereplay";

export default function RaceReplay() {
  const [searchParams] = useSearchParams();
  const canvasRef = useRef(null);

  // State management (unchanged)
  const [currentLap, setCurrentLap] = useState(0);
  const [frameIndex, setFrameIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [selectedDriver, setSelectedDriver] = useState(null);
  const [showDRS, setShowDRS] = useState(true);
  const [showTelemetry, setShowTelemetry] = useState(true);
  const [rightPanelTab, setRightPanelTab] = useState("standings");
  const [focusMode, setFocusMode] = useState(false);
  const [realTimeMode, setRealTimeMode] = useState(false);

  // Real-time sync ref
  const syncStartRef = useRef(null); // { wallMs, raceTime }

  const raceNum = searchParams.get("race") || 21;

  // Data fetching
  const { raceData, loading, error } = useRaceReplayData({ raceNum });
  const { lapPredictions, predictionsLoading } = useLapPredictions({ raceNum });

  // Derived frame info
  const frameIdx = raceData
    ? Math.min(Math.floor(frameIndex), raceData.frames.length - 1)
    : 0;

  const currentFrame = raceData?.frames?.[frameIdx] ?? null;
  const totalLaps = raceData?.frames?.[raceData.frames.length - 1]?.lap ?? null;

  // TrackRenderer frame change callback
  const handleFrameChange = (idx) => {
    setFrameIndex(idx);
    if (raceData && idx >= raceData.frames.length - 1) setIsPlaying(false);
  };

  // Keep lap in sync
  useEffect(() => {
    if (currentFrame && currentFrame.lap) setCurrentLap(currentFrame.lap);
  }, [currentFrame]);

  // Events + smoothed gaps
  const { raceEvents, smoothedGapsRef } = useRaceEvents({ currentFrame, frameIdx });

  // Keyboard shortcuts
  useKeyboardShortcuts({
    raceData,
    currentFrame,
    isPlaying,
    setIsPlaying,
    setFrameIndex,
    showDRS,
    setShowDRS,
    showTelemetry,
    setShowTelemetry,
    focusMode,
    setFocusMode,
    realTimeMode,
    setRealTimeMode,
    setPlaybackSpeed,
    syncStartRef,
  });

  // Real-time sync loop
  useRealTimeSync({
    realTimeMode,
    raceData,
    setFrameIndex,
    setRealTimeMode,
    syncStartRef,
  });

  // Predictions resolve (ML first, fallback heuristic)
  const displayPredictions = useMemo(() => {
    const mlPreds = lapPredictions?.[String(currentLap)];
    if (mlPreds && mlPreds.length > 0) {
      return mlPreds.slice(0, 10).map((p) => ({
        code: p.driver_code,
        name: p.driver_name,
        team: p.team,
        probability: Math.round(p.confidence),
        position: p.predicted_position,
        tireCompound: "MEDIUM",
        tireAge: 0,
        gap: "",
      }));
    }
    return computePredictions(currentFrame);
  }, [lapPredictions, currentLap, currentFrame]);

  // Loading / Error UI (identiek gedrag)
  if (loading) {
    return (
      <div className="space-y-6">
        <Card className="p-8" clip>
          <div className="flex items-center gap-4">
            <div
              className="h-10 w-10 rounded-full border-4 border-black/10 dark:border-white/10 animate-spin"
              style={{ borderTopColor: "rgb(var(--accent))" }}
            />
            <div>
              <div className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">
                Loading race replay
              </div>
              <div className="mt-1 text-sm text-neutral-500 dark:text-neutral-400">
                Fetching frame data from the server…
              </div>
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
                {error ? "Error loading replay" : "No data available"}
              </div>
              <div className="mt-1 text-sm text-neutral-500 dark:text-neutral-400">
                {error || "Could not load race data for replay."}
              </div>
            </div>
            <Badge variant="danger">Error</Badge>
          </div>
          <button
            type="button"
            onClick={() => window.location.reload()}
            className={[
              "mt-4 inline-flex items-center justify-center",
              "rounded-2xl px-4 py-2 text-sm font-semibold",
              "bg-transparent",
              "ring-1 ring-black/5 dark:ring-white/10",
              "text-neutral-700 dark:text-neutral-200",
              "hover:bg-black/[0.03] dark:hover:bg-white/[0.05]",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[rgb(var(--accent))]",
              "active:scale-[0.99] transition",
            ].join(" ")}
          >
            Retry
          </button>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {focusMode ? (
        <FocusOverlay
          raceData={raceData}
          canvasRef={canvasRef}
          frameIndex={frameIndex}
          setFrameIndex={setFrameIndex}
          handleFrameChange={handleFrameChange}
          currentFrame={currentFrame}
          currentLap={currentLap}
          totalLaps={totalLaps}
          isPlaying={isPlaying}
          setIsPlaying={setIsPlaying}
          playbackSpeed={playbackSpeed}
          setPlaybackSpeed={setPlaybackSpeed}
          showDRS={showDRS}
          setShowDRS={setShowDRS}
          showTelemetry={showTelemetry}
          setShowTelemetry={setShowTelemetry}
          focusMode={focusMode}
          setFocusMode={setFocusMode}
          realTimeMode={realTimeMode}
          setRealTimeMode={setRealTimeMode}
          selectedDriver={selectedDriver}
          setSelectedDriver={setSelectedDriver}
          predictionsLoading={predictionsLoading}
          lapPredictions={lapPredictions}
          displayPredictions={displayPredictions}
          syncStartRef={syncStartRef}
          smoothedGaps={smoothedGapsRef.current}
        />
      ) : (
        <NormalView
          raceData={raceData}
          canvasRef={canvasRef}
          frameIndex={frameIndex}
          setFrameIndex={setFrameIndex}
          handleFrameChange={handleFrameChange}
          currentFrame={currentFrame}
          currentLap={currentLap}
          totalLaps={totalLaps}
          isPlaying={isPlaying}
          setIsPlaying={setIsPlaying}
          playbackSpeed={playbackSpeed}
          setPlaybackSpeed={setPlaybackSpeed}
          showDRS={showDRS}
          setShowDRS={setShowDRS}
          showTelemetry={showTelemetry}
          setShowTelemetry={setShowTelemetry}
          focusMode={focusMode}
          realTimeMode={realTimeMode}
          setRealTimeMode={setRealTimeMode}
          selectedDriver={selectedDriver}
          setSelectedDriver={setSelectedDriver}
          rightPanelTab={rightPanelTab}
          setRightPanelTab={setRightPanelTab}
          raceEvents={raceEvents}
          syncStartRef={syncStartRef}
          smoothedGaps={smoothedGapsRef.current}
          setFocusMode={setFocusMode}
        />
      )}
    </div>
  );
}