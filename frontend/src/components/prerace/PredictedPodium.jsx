// src/components/prerace/PredictedPodium.jsx
import { useMemo } from "react";
import Card from "../ui/Card";
import { getTeamColor } from "../../utils/teamColors";

function clamp01(n) {
  const x = Number(n);
  if (!Number.isFinite(x)) return 0;
  return Math.min(100, Math.max(0, x));
}

function toInt(n, fallback = null) {
  const x = Number(n);
  return Number.isFinite(x) ? Math.round(x) : fallback;
}

function formatDeltaVsGrid(delta) {
  if (delta == null) return "Δ —";
  if (delta > 0) return `▲ ${delta} vs grid`;
  if (delta < 0) return `▼ ${Math.abs(delta)} vs grid`;
  return "—";
}

export default function PredictedPodium({ predictions = [] }) {
  const podium = useMemo(() => {
    const top = Array.isArray(predictions) ? predictions.slice(0, 3) : [];
    return [
      { pred: top?.[0], pos: 1 },
      { pred: top?.[1], pos: 2 },
      { pred: top?.[2], pos: 3 },
    ];
  }, [predictions]);

  return (
    // Match PredictionsPanel outer structure (without nav buttons)
    <div className="flex h-full min-h-0 flex-col gap-3">
      <div className="grid min-h-0 flex-1 grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {podium.map(({ pred, pos }) => {
          if (!pred) return null;

          const driver = pred.driver_name || pred.driver || "Unknown";
          const team = pred.team ?? "";
          const teamColor = getTeamColor(team) || "rgba(0,0,0,0.12)";

          const confidence = clamp01(pred.confidence ?? 0);

          const grid = toInt(pred.grid_position, null);
          const delta = grid == null ? null : grid - pos; // positive = improved vs grid

          return (
            <Card key={pos} className="relative p-5" clip>
              {/* Same top accent line */}
              <div
                className="absolute left-0 top-0 h-1 w-full"
                style={{ background: teamColor }}
                aria-hidden="true"
              />

              {/* Same header anatomy */}
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="text-[11px] font-semibold uppercase tracking-widest text-neutral-500 dark:text-neutral-400">
                    Prediction
                  </div>

                  <div className="mt-2 truncate text-lg font-semibold tracking-tight text-neutral-900 dark:text-neutral-50">
                    {driver}
                  </div>

                  {team ? (
                    <div className="mt-1 truncate text-sm text-neutral-500 dark:text-neutral-400">
                      {team}
                    </div>
                  ) : null}
                </div>

                {/* Same position pill */}
                <div
                  className={[
                    "shrink-0 rounded-full px-3 py-1.5",
                    "text-sm font-semibold tabular-nums",
                    "bg-neutral-100 text-neutral-900 ring-1 ring-black/10",
                    "dark:bg-white/10 dark:text-neutral-50 dark:ring-white/10",
                  ].join(" ")}
                  title={`Predicted P${pos}`}
                >
                  P{pos}
                </div>
              </div>

              {/* Same confidence row */}
              <div className="mt-4 flex items-baseline justify-between gap-3">
                <div className="text-sm text-neutral-600 dark:text-neutral-400">
                  Confidence
                </div>
                <div className="text-base font-semibold tabular-nums text-neutral-900 dark:text-neutral-50">
                  {Math.round(confidence)}%
                </div>
              </div>

              {/* Same bar */}
              <div className="mt-3">
                <div className="h-2 w-full overflow-hidden rounded-full bg-neutral-200/70 dark:bg-white/10">
                  <div
                    className="h-full rounded-full"
                    style={{ width: `${confidence}%`, background: teamColor }}
                    aria-hidden="true"
                  />
                </div>
              </div>

              {/* Bottom meta row: match PredictionsPanel density */}
              <div className="mt-4 flex items-center justify-between text-[11px] text-neutral-600 dark:text-neutral-400">
                <span className="tabular-nums">
                  {grid == null ? "Grid —" : `Grid P${grid}`}
                </span>

                <span
                  className={[
                    "tabular-nums",
                    delta == null
                      ? ""
                      : delta > 0
                        ? "font-semibold text-emerald-600 dark:text-emerald-400"
                        : delta < 0
                          ? "font-semibold text-red-600 dark:text-red-400"
                          : "text-neutral-500 dark:text-neutral-500",
                  ].join(" ")}
                  title="Change versus grid position"
                >
                  {formatDeltaVsGrid(delta)}
                </span>
              </div>
            </Card>
          );
        })}

        {/* Keep identical empty-grid behavior as PredictionsPanel (optional but helps layout stability) */}
        {podium.filter((x) => x.pred).length < 3
          ? Array.from({ length: 3 - podium.filter((x) => x.pred).length }).map(
              (_, i) => <div key={`pad-${i}`} className="hidden lg:block" />
            )
          : null}

        {podium.filter((x) => x.pred).length < 2 ? (
          <div className="hidden sm:block lg:hidden" />
        ) : null}
      </div>
    </div>
  );
}