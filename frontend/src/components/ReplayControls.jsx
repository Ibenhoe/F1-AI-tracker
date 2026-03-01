import React from 'react';

const SPEEDS = [0.25, 0.5, 1, 2, 4];

const TogglePill = ({ active, onClick, title, children }) => (
  <button
    type="button"
    onClick={onClick}
    title={title}
    className={[
      'rounded-full border px-3 py-1 text-xs font-semibold transition select-none',
      active
        ? 'border-red-500 bg-red-500/15 text-red-500 dark:border-red-400 dark:text-red-400'
        : 'border-neutral-200 bg-white text-neutral-500 hover:border-neutral-400 hover:text-neutral-700 dark:border-neutral-800 dark:bg-neutral-950/40 dark:text-neutral-400 dark:hover:border-neutral-600',
    ].join(' ')}
  >
    {children}
  </button>
);

/**
 * ReplayControls — Tailwind-native playback bar
 */
const ReplayControls = ({
  isPlaying,
  playbackSpeed,
  currentFrame,
  totalFrames,
  showDRS,
  showTelemetry,
  focusMode,
  realTimeMode,
  onPlayPause,
  onSpeedChange,
  onFrameChange,
  onDRSToggle,
  onTelemetryToggle,
  onFocusToggle,
  onRealTimeToggle,
}) => {
  const pct = totalFrames > 0 ? (Math.floor(currentFrame) / totalFrames) * 100 : 0;
  const remaining = totalFrames - Math.floor(currentFrame);

  return (
    <div className="flex flex-col gap-3">
      {/* ── Scrubber ── */}
      <div className="group relative h-4 cursor-pointer">
        {/* track */}
        <div className="absolute inset-y-[5px] left-0 right-0 overflow-hidden rounded-full bg-neutral-200 dark:bg-neutral-800">
          <div
            className="h-full rounded-full bg-red-500 transition-[width] duration-75"
            style={{ width: `${pct}%` }}
          />
        </div>
        {/* invisible range input on top */}
        <input
          type="range"
          min={0}
          max={totalFrames}
          value={Math.floor(currentFrame)}
          onChange={(e) => onFrameChange(Number(e.target.value))}
          className="absolute inset-0 h-full w-full cursor-pointer opacity-0"
          title="Seek"
        />
        {/* thumb dot (cosmetic) */}
        <div
          className="pointer-events-none absolute top-1/2 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-red-500 bg-white shadow transition-opacity dark:bg-neutral-900"
          style={{ left: `${pct}%` }}
        />
      </div>

      {/* ── Controls row ── */}
      <div className="flex flex-wrap items-center gap-2">
        {/* Play / Pause */}
        <button
          type="button"
          onClick={onPlayPause}
          title={isPlaying ? 'Pause (SPACE)' : 'Play (SPACE)'}
          className="flex h-9 w-9 items-center justify-center rounded-full border border-neutral-200 bg-white text-base text-neutral-900 shadow-sm transition hover:bg-neutral-50 dark:border-neutral-700 dark:bg-neutral-900 dark:text-neutral-100 dark:hover:bg-neutral-800"
        >
          {isPlaying ? '⏸' : '▶'}
        </button>

        {/* Speed pill group */}
        <div className="flex items-center gap-1 rounded-full border border-neutral-200 bg-neutral-100/60 px-1 py-0.5 dark:border-neutral-800 dark:bg-neutral-900/60">
          {SPEEDS.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => onSpeedChange(s)}
              className={[
                'rounded-full px-2.5 py-0.5 text-xs font-semibold transition',
                playbackSpeed === s
                  ? 'bg-neutral-900 text-white dark:bg-neutral-100 dark:text-neutral-900'
                  : 'text-neutral-600 hover:text-neutral-900 dark:text-neutral-400 dark:hover:text-neutral-100',
              ].join(' ')}
            >
              {s}×
            </button>
          ))}
        </div>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Frame / time remaining */}
        <div className="hidden sm:flex flex-col items-end text-xs tabular-nums text-neutral-500 dark:text-neutral-400">
          <span>{Math.floor(currentFrame).toLocaleString()} / {totalFrames.toLocaleString()}</span>
          <span>~{formatFrames(remaining)} left</span>
        </div>

        {/* Divider */}
        <div className="h-5 w-px bg-neutral-200 dark:bg-neutral-800" />

        {/* Toggle pills */}
        <TogglePill active={showDRS} onClick={onDRSToggle} title="Toggle DRS zones (D)">DRS</TogglePill>
        <TogglePill active={showTelemetry} onClick={onTelemetryToggle} title="Toggle telemetry (T)">TEL</TogglePill>
        <TogglePill active={focusMode} onClick={onFocusToggle} title="Focus mode + AI predictions (F)">FOCUS</TogglePill>
        <TogglePill active={realTimeMode} onClick={onRealTimeToggle} title="Sync replay to real race speed (S)">SYNC</TogglePill>

        {/* Fullscreen */}
        <button
          type="button"
          title="Fullscreen"
          onClick={() => document.documentElement.requestFullscreen?.()}
          className="flex h-7 w-7 items-center justify-center rounded border border-neutral-200 text-neutral-500 transition hover:border-neutral-400 hover:text-neutral-700 dark:border-neutral-800 dark:text-neutral-400 dark:hover:border-neutral-600"
        >
          ⛶
        </button>
      </div>
    </div>
  );
};

function formatFrames(frames) {
  if (frames <= 0) return '0s';
  // roughly 2 frames per second of real race time at 1× speed
  const seconds = Math.ceil(frames / 2);
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

export default ReplayControls;
