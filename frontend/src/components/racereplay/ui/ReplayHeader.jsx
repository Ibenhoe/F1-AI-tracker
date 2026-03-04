// src/components/racereplay/ui/ReplayHeader.jsx
import React, { useMemo } from "react";
import Badge from "../../ui/Badge";
import formatTime from "../utils/formatTime";

function formatSpeed(x) {
  const n = Number(x || 1);
  if (!Number.isFinite(n)) return "1x";
  // avoid 1.00x etc
  return n % 1 === 0 ? `${n}x` : `${n}x`;
}

export default function ReplayHeader({
  raceData,
  currentLap,
  currentFrame,
  isPlaying,
  playbackSpeed,
  realTimeMode = false,
}) {
  const title = raceData?.raceName || "Race Replay";

  const timeLabel = useMemo(() => {
    if (!currentFrame) return null;
    return formatTime(currentFrame.raceTime);
  }, [currentFrame]);

  const status = realTimeMode ? "Synced" : isPlaying ? "Playing" : "Paused";

  return (
    <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
      <div className="space-y-1 min-w-0">
        <h1 className="text-2xl font-semibold tracking-tight text-neutral-900 dark:text-neutral-50 truncate">
          {title}
        </h1>
        <p className="text-sm text-neutral-500 dark:text-neutral-400">
          Full-session playback with live standings, events, and telemetry overlays.
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <div
          className={[
            "flex flex-wrap items-center gap-2",
            "rounded-2xl px-3 py-2",
            "bg-white dark:bg-neutral-950/40",
            "ring-1 ring-black/5 dark:ring-white/10",
          ].join(" ")}
        >
          <Badge variant="accent">Replay</Badge>

          <Badge variant="neutral">
            LAP <span className="tabular-nums">{currentLap || 0}</span>
          </Badge>

          {timeLabel ? (
            <Badge variant="neutral">
              <span className="tabular-nums">{timeLabel}</span>
            </Badge>
          ) : null}

          <Badge
            variant={
              realTimeMode ? "accent" : isPlaying ? "warning" : "neutral"
            }
          >
            {status}
          </Badge>

          <Badge variant="neutral">
            <span className="tabular-nums">{formatSpeed(playbackSpeed)}</span>
          </Badge>
        </div>
      </div>
    </div>
  );
}