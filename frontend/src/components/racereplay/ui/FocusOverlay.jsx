// src/components/racereplay/ui/FocusOverlay.jsx
import React from "react";
import Card from "../../ui/Card";
import Badge from "../../ui/Badge";
import PredictionsPanel from "./PredictionsPanel";

// Existing components (kept where they are now)
import TrackRenderer from "../../TrackRenderer";
import ReplayControls from "../../ReplayControls";
import ReplayLeaderboard from "../../ReplayLeaderboard";

export default function FocusOverlay({
  raceData,
  canvasRef,

  frameIndex,
  setFrameIndex,
  handleFrameChange,

  currentFrame,
  currentLap,
  totalLaps,

  isPlaying,
  setIsPlaying,
  playbackSpeed,
  setPlaybackSpeed,

  showDRS,
  setShowDRS,
  showTelemetry,
  setShowTelemetry,

  focusMode,
  setFocusMode,

  realTimeMode,
  setRealTimeMode,

  selectedDriver,
  setSelectedDriver,

  predictionsLoading,
  lapPredictions,
  displayPredictions,

  syncStartRef,
  smoothedGaps,
}) {
  return (
    <div
      className={[
        "fixed inset-0 z-50",
        "flex flex-col",
        "bg-white text-neutral-900",
        "dark:bg-neutral-950 dark:text-neutral-100",
      ].join(" ")}
    >
      {/* TRACK (full screen) */}
      <div className="relative flex-1 overflow-hidden">
        {/* Track surface */}
        <div className="absolute inset-0">
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
            smoothedGaps={smoothedGaps}
          />
        </div>

        {/* Floating leaderboard – left */}
        <Card
          className={[
            "absolute left-4 top-4 z-20",
            "w-72",
            "p-3",
            "max-h-[calc(100vh-180px)]",
            "flex flex-col",
            "bg-white/80 dark:bg-neutral-950/50",
            "ring-1 ring-black/5 dark:ring-white/10",
            "backdrop-blur-xl",
          ].join(" ")}
          clip
        >
          <div
            className="min-h-0 flex-1 overflow-y-auto scrollbar-hide"
            style={{ WebkitOverflowScrolling: "touch" }}
          >
            <ReplayLeaderboard
              drivers={currentFrame?.drivers || []}
              selectedDriver={selectedDriver}
              onDriverSelect={setSelectedDriver}
              currentLap={currentLap}
              totalLaps={totalLaps}
            />
          </div>
        </Card>

        {/* Floating AI predictions panel – right side (keep component) */}
        <PredictionsPanel
          displayPredictions={displayPredictions}
          predictionsLoading={predictionsLoading}
          lapPredictions={lapPredictions}
          currentLap={currentLap}
          totalLaps={totalLaps}
        />

        {/* Real-time sync badge (top center) */}
        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-30">
          {realTimeMode ? (
            <div
              className={[
                "inline-flex items-center gap-2 rounded-full px-3 py-1.5",
                "bg-red-500/10 dark:bg-red-500/10",
                "ring-1 ring-red-500/30",
                "backdrop-blur-md",
              ].join(" ")}
            >
              <span className="inline-block h-1.5 w-1.5 rounded-full bg-red-500 animate-pulse" />
              <span className="text-[11px] font-semibold tracking-widest text-red-700 dark:text-red-300">
                LIVE SYNC
              </span>
            </div>
          ) : (
            <Badge variant="neutral">Focus mode</Badge>
          )}
        </div>
      </div>

      {/* CONTROLS (bottom bar) */}
      <div
        className={[
          "relative z-40",
          "border-t border-black/5 dark:border-white/10",
          "bg-white/70 dark:bg-neutral-950/60",
          "ring-1 ring-black/5 dark:ring-white/10",
          "backdrop-blur-xl",
          "p-3 sm:p-4",
        ].join(" ")}
      >
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
                wallMs: performance.now(),
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
              syncStartRef.current = {
                wallMs: performance.now(),
                raceTime: currentFrame.raceTime || 0,
              };
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
  );
}