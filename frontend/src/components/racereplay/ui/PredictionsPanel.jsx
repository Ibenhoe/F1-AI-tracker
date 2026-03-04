// src/components/racereplay/ui/PredictionsPanel.jsx
import React, { useMemo } from "react";
import Card from "../../ui/Card";
import Badge from "../../ui/Badge";
import { getTeamColor } from "../../../utils/teamColors";

function clamp01(n) {
  const x = Number(n);
  if (!Number.isFinite(x)) return 0;
  return Math.max(0, Math.min(100, x));
}

export default function PredictionsPanel({
  displayPredictions = [],
  predictionsLoading,
  lapPredictions,
  currentLap,
  totalLaps,
}) {
  const metaLabel = useMemo(() => {
    const isML = lapPredictions && !predictionsLoading;
    return isML ? "Real ML model" : "Heuristic";
  }, [lapPredictions, predictionsLoading]);

  return (
    <Card
      className={[
        "absolute right-4 top-4 z-20 w-60",
        "bg-white/80 dark:bg-neutral-950/50",
        "backdrop-blur-md",
        "ring-1 ring-black/5 dark:ring-white/10",
      ].join(" ")}
      clip
    >
      {/* Header */}
      <div className="px-4 py-3 border-b border-black/5 dark:border-white/10">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="text-[11px] font-semibold uppercase tracking-widest text-neutral-600 dark:text-neutral-300">
              AI Win Prediction
            </p>
            <p className="mt-1 text-[11px] text-neutral-500 dark:text-neutral-400">
              {metaLabel} · Lap{" "}
              <span className="tabular-nums">{currentLap}</span>
              {totalLaps ? (
                <>
                  {" "}
                  / <span className="tabular-nums">{totalLaps}</span>
                </>
              ) : null}
            </p>
          </div>

          <div className="flex items-center gap-2">
            {predictionsLoading ? (
              <div className="h-4 w-4 rounded-full border-2 border-black/10 dark:border-white/20 border-t-black/40 dark:border-t-white/60 animate-spin" />
            ) : (
              <Badge variant="neutral">Top 10</Badge>
            )}
          </div>
        </div>
      </div>

      {/* List */}
      <div className="max-h-[calc(100vh-180px)] overflow-y-auto px-3 py-3 space-y-2">
        {Array.isArray(displayPredictions) && displayPredictions.length > 0 ? (
          displayPredictions.map((p, idx) => {
            const prob = clamp01(p?.probability);
            const teamColor = getTeamColor(p?.team) || "rgb(var(--accent))";

            return (
              <div
                key={`${p?.code || "X"}-${idx}`}
                className={[
                  "rounded-2xl px-3 py-2",
                  "bg-black/[0.03] dark:bg-white/[0.06]",
                  "ring-1 ring-black/5 dark:ring-white/10",
                ].join(" ")}
              >
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2 min-w-0">
                    <span className="w-5 text-right text-[11px] tabular-nums text-neutral-500 dark:text-neutral-400">
                      {idx + 1}
                    </span>

                    <div className="min-w-0">
                      <div className="flex items-center gap-2 min-w-0">
                        <span className="truncate text-sm font-semibold text-neutral-900 dark:text-neutral-100">
                          {p?.code || "—"}
                        </span>
                        <span
                          className="h-1.5 w-1.5 rounded-full opacity-90"
                          style={{ background: teamColor }}
                          aria-hidden="true"
                        />
                      </div>
                      {p?.team ? (
                        <div className="truncate text-[11px] text-neutral-500 dark:text-neutral-400">
                          {p.team}
                        </div>
                      ) : null}
                    </div>
                  </div>

                  <span
                    className="shrink-0 text-sm font-semibold tabular-nums"
                    style={{ color: teamColor }}
                  >
                    {Math.round(prob)}%
                  </span>
                </div>

                <div className="mt-2 h-1.5 w-full rounded-full bg-black/10 dark:bg-white/10 overflow-hidden">
                  <div
                    className="h-full rounded-full transition-[width] duration-300 ease-out"
                    style={{
                      width: `${prob}%`,
                      background: teamColor,
                    }}
                  />
                </div>
              </div>
            );
          })
        ) : (
          <div className="px-2 py-6 text-center">
            <p className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">No predictions</p>
            <p className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">
              Start playback to generate predictions.
            </p>
          </div>
        )}
      </div>
    </Card>
  );
}