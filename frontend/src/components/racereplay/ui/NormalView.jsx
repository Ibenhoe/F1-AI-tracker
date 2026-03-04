// src/components/racereplay/ui/NormalView.jsx
import React from "react";
import Card from "../../ui/Card";
import ReplayHeader from "./ReplayHeader";
import RightPanel from "./RightPanel";

// Existing components (kept where they are now)
import TrackRenderer from "../../TrackRenderer";
import ReplayControls from "../../ReplayControls";

export default function NormalView({
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

    rightPanelTab,
    setRightPanelTab,
    raceEvents,

    syncStartRef,
    smoothedGaps,
}) {
    return (
        <>
            {/* HEADER */}
            <ReplayHeader
                raceData={raceData}
                currentLap={currentLap}
                currentFrame={currentFrame}
                isPlaying={isPlaying}
                playbackSpeed={playbackSpeed}
                realTimeMode={realTimeMode}
            />

            {/* TOP BAR CONTROLS (moved up) */}
            <Card className="p-5" clip>
                <div className="min-h-0">
                    <ReplayControls
                        isPlaying={realTimeMode ? false : isPlaying}
                        playbackSpeed={playbackSpeed}
                        currentFrame={frameIndex}
                        totalFrames={raceData?.frames?.length || 0}
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
            </Card>

            {/* TRACK (FULL WIDTH) */}
            <Card className="p-5" clip>
                <div className="w-full">
                    <div className="h-[340px] sm:h-[420px] lg:h-[520px] xl:h-[620px] w-full">
                        <div className="h-full w-full">
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
                                smoothedGaps={smoothedGaps}
                            />
                        </div>
                    </div>
                </div>
            </Card>

            {/* PANELS (FULL WIDTH UNDER TRACK) */}
            <RightPanel
                rightPanelTab={rightPanelTab}
                setRightPanelTab={setRightPanelTab}
                currentFrame={currentFrame}
                selectedDriver={selectedDriver}
                setSelectedDriver={setSelectedDriver}
                currentLap={currentLap}
                totalLaps={totalLaps}
                showTelemetry={showTelemetry}
                raceEvents={raceEvents}
            />

            {/* KEYBOARD SHORTCUTS */}
<Card className="p-5" clip>
  <div className="mb-4">
    <div className="text-[11px] font-semibold uppercase tracking-widest text-neutral-500 dark:text-neutral-400">
      Keyboard
    </div>
    <div className="mt-1 text-sm font-semibold text-neutral-900 dark:text-neutral-50">
      Shortcuts
    </div>
  </div>

  <div className="rounded-2xl ring-1 ring-black/5 dark:ring-white/10 overflow-hidden">
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 divide-y sm:divide-y-0 sm:divide-x divide-black/5 dark:divide-white/10">

      {[
        { k: "SPACE", t: "Play / Pause" },
        { k: "← / →", t: "Seek" },
        { k: "+ / −", t: "Speed" },
        { k: "D", t: "DRS" },
        { k: "T", t: "Telemetry" },
        { k: "F", t: "Focus + Predictions" },
        { k: "S", t: "Sync to real race" },
        { k: "R", t: "Reset" },
      ].map(({ k, t }) => (
        <div
          key={k}
          className="flex items-center gap-4 px-4 py-3 bg-white dark:bg-neutral-950/40"
        >
          <kbd
            className={[
              "flex h-8 min-w-[44px] items-center justify-center",
              "rounded-xl px-2",
              "font-mono text-[11px] font-bold",
              "bg-black/[0.03] dark:bg-white/[0.06]",
              "ring-1 ring-black/5 dark:ring-white/10",
              "text-neutral-800 dark:text-neutral-200",
            ].join(" ")}
          >
            {k}
          </kbd>

          <span className="text-sm font-medium text-neutral-800 dark:text-neutral-200">
            {t}
          </span>
        </div>
      ))}
    </div>
  </div>
</Card>
        </>
    );
}