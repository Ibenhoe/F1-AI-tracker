// src/components/ReplayControls.jsx
import React, { useMemo } from "react";
import SegmentedControl from "./racereplay/ui/controls/SegmentedControl";

const SPEEDS = [0.25, 0.5, 1, 2, 4];

function formatFrames(frames) {
  if (frames <= 0) return "0s";
  // Assumption from your existing code: ~2 frames per second
  const seconds = Math.ceil(frames / 2);
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

const ICONS = {
  play: (
    <svg viewBox="0 0 24 24" className="h-4 w-4" aria-hidden="true">
      <path fill="currentColor" d="M8 5v14l11-7z" />
    </svg>
  ),
  pause: (
    <svg viewBox="0 0 24 24" className="h-4 w-4" aria-hidden="true">
      <path fill="currentColor" d="M6 5h4v14H6zm8 0h4v14h-4z" />
    </svg>
  ),
  fullscreen: (
    <svg viewBox="0 0 24 24" className="h-4 w-4" aria-hidden="true">
      <path
        fill="currentColor"
        d="M7 14H5v5h5v-2H7zm0-4h2V7h3V5H5v5zm10 9h-3v2h5v-5h-2zm0-14V5h-5v2h3v3z"
      />
    </svg>
  ),
};

function TogglePill({ active, onClick, title, children }) {
  const accent = "rgb(var(--accent))";

  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      className={[
        "rounded-full px-3 py-1 text-xs font-semibold transition select-none",
        active ? "ring-1 ring-[rgba(var(--accent),0.35)]" : "ring-1 ring-black/5 dark:ring-white/10",
        "bg-white dark:bg-neutral-950/40",
        "hover:bg-black/[0.02] dark:hover:bg-white/[0.03]",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[rgb(var(--accent))] focus-visible:ring-offset-2 focus-visible:ring-offset-transparent",
        active ? "text-neutral-900 dark:text-neutral-50" : "text-neutral-500 dark:text-neutral-400",
      ].join(" ")}
      style={
        active
          ? {
            background: "rgba(var(--accent),0.10)",
            color: accent,
          }
          : undefined
      }
    >
      {children}
    </button>
  );
}

/**
 * ReplayControls — themed playback bar (no CSS)
 * - No emojis
 * - Responsive: stacks on small screens, aligns on large screens
 */
export default function ReplayControls({
  isPlaying,
  playbackSpeed,
  currentFrame,
  totalFrames,
  realTimeMode,
  onPlayPause,
  onSpeedChange,
  onFrameChange,
  onRealTimeToggle,
}) {
  const accent = "rgb(var(--accent))";

  const pct = totalFrames > 0 ? (Math.floor(currentFrame) / totalFrames) * 100 : 0;
  const remaining = totalFrames - Math.floor(currentFrame);

  const speedItems = useMemo(
    () =>
      SPEEDS.map((s) => ({
        id: String(s),
        label: `${s}×`,
      })),
    []
  );

  return (
    <div className="flex flex-col gap-4">
      {/* Scrubber */}
      <div className="group relative h-5">
        {/* Track */}
        <div
          className={[
            "absolute inset-y-[7px] left-0 right-0 overflow-hidden rounded-full",
            "bg-black/[0.06] dark:bg-white/[0.08]",
            "ring-1 ring-black/5 dark:ring-white/10",
          ].join(" ")}
        >
          <div
            className="h-full rounded-full transition-[width] duration-75"
            style={{ width: `${pct}%`, background: accent }}
          />
        </div>

        {/* Input */}
        <input
          type="range"
          min={0}
          max={Math.max(0, totalFrames)}
          value={Math.floor(currentFrame)}
          onChange={(e) => onFrameChange(Number(e.target.value))}
          className="absolute inset-0 h-full w-full cursor-pointer opacity-0"
          title="Seek"
        />

        {/* Thumb */}
        <div
          className={[
            "pointer-events-none absolute top-1/2 h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full",
            "bg-white dark:bg-neutral-950",
            "ring-1 ring-black/10 dark:ring-white/10",
            "shadow-sm",
          ].join(" ")}
          style={{ left: `${pct}%`, border: `2px solid ${accent}` }}
        />
      </div>

      {/* Controls row */}
      <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:gap-2">
        {/* Left cluster (play + speed) */}
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:gap-2 lg:flex-1">
          <button
            type="button"
            onClick={onPlayPause}
            title={isPlaying ? "Pause (SPACE)" : "Play (SPACE)"}
            className={[
              "flex h-9 w-9 items-center justify-center rounded-full",
              "bg-white dark:bg-neutral-950/40",
              "ring-1 ring-black/5 dark:ring-white/10",
              "text-neutral-900 dark:text-neutral-50",
              "transition hover:bg-black/[0.02] dark:hover:bg-white/[0.03]",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[rgb(var(--accent))] focus-visible:ring-offset-2 focus-visible:ring-offset-transparent",
            ].join(" ")}
            style={isPlaying ? { color: accent } : undefined}
          >
            <span className="sr-only">{isPlaying ? "Pause" : "Play"}</span>
            {isPlaying ? ICONS.pause : ICONS.play}
          </button>

          <div className="w-full sm:w-auto sm:min-w-[260px]">
            <SegmentedControl
              value={String(playbackSpeed * 4)}
              onChange={(id) => onSpeedChange(Number(id) / 4)}
              ariaLabel="Replay speed"
              items={speedItems}
            />
          </div>
        </div>

        {/* Right cluster (status + toggles + fullscreen) */}
        <div className="flex flex-wrap items-center gap-2 lg:justify-end">
          {/* Frame / remaining */}
          <div className="hidden sm:flex flex-col items-end text-xs tabular-nums text-neutral-500 dark:text-neutral-400">
            <span>
              {Math.floor(currentFrame).toLocaleString()} / {totalFrames.toLocaleString()}
            </span>
            <span>~{formatFrames(remaining)} left</span>
          </div>

          {/* Divider */}
          <div className="hidden sm:block h-5 w-px bg-black/5 dark:bg-white/10" />

          {/* Toggle pills */}
          <TogglePill active={realTimeMode} onClick={onRealTimeToggle} title="Sync replay to real race speed (S)">
            SYNC
          </TogglePill>

          <button
            type="button"
            title="Fullscreen"
            onClick={() => document.documentElement.requestFullscreen?.()}
            className={[
              "flex h-9 w-9 items-center justify-center rounded-full",
              "bg-white dark:bg-neutral-950/40",
              "ring-1 ring-black/5 dark:ring-white/10",
              "text-neutral-500 dark:text-neutral-400",
              "transition hover:bg-black/[0.02] dark:hover:bg-white/[0.03]",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[rgb(var(--accent))] focus-visible:ring-offset-2 focus-visible:ring-offset-transparent",
            ].join(" ")}
          >
            <span className="sr-only">Fullscreen</span>
            {ICONS.fullscreen}
          </button>
        </div>
      </div>
    </div>
  );
}