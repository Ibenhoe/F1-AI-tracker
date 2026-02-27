// src/components/prerace/PredictedPodium.jsx
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

export default function PredictedPodium({ predictions = [] }) {
  const top10 = Array.isArray(predictions) ? predictions.slice(0, 10) : [];

  // Keep podium ordering + vertical offsets
  const podium = [
    { pred: top10?.[1], pos: 2, offset: "pt-6" },
    { pred: top10?.[0], pos: 1, offset: "pt-0" },
    { pred: top10?.[2], pos: 3, offset: "pt-10" },
  ];

  return (
    <div className="grid grid-cols-3 gap-4">
      {podium.map(({ pred, pos, offset }) => {
        // Empty placeholder keeps spacing if data missing
        if (!pred) return <div key={pos} className={offset} />;

        const driver = pred.driver_name || pred.driver || "Unknown";
        const team = pred.team ?? "—";
        const teamColor = getTeamColor(team) || "rgba(0,0,0,0.12)";

        const confidence = clamp01(pred.confidence ?? 0);

        // backend grid_position might be number-like string
        const grid = toInt(pred.grid_position, null);
        const predicted = pos;
        const delta = grid == null ? null : grid - predicted; // positive = improved vs grid

        return (
          <div key={pos} className={offset}>
            <Card className="relative p-5" clip>
              {/* Same as PredictionsPanel: top accent line */}
              <div
                className="absolute left-0 top-0 h-1 w-full"
                style={{ background: teamColor }}
                aria-hidden="true"
              />

              {/* Header row */}
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

                {/* Position pill (same style as PredictionsPanel) */}
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

              {/* Confidence */}
              <div className="mt-4 flex items-baseline justify-between gap-3">
                <div className="text-sm text-neutral-600 dark:text-neutral-400">
                  Confidence
                </div>
                <div className="text-base font-semibold tabular-nums text-neutral-900 dark:text-neutral-50">
                  {Math.round(confidence)}%
                </div>
              </div>

              <div className="mt-3">
                <div className="h-2 w-full overflow-hidden rounded-full bg-neutral-200/70 dark:bg-white/10">
                  <div
                    className="h-full rounded-full"
                    style={{ width: `${confidence}%`, background: teamColor }}
                    aria-hidden="true"
                  />
                </div>
              </div>

              {/* Bottom meta row (match PredictionsPanel density) */}
              <div className="mt-4 flex items-center justify-between text-[11px] text-neutral-600 dark:text-neutral-400">
                <span className="tabular-nums">
                  {grid == null ? "Grid —" : `Grid P${grid}`}
                </span>

                {delta == null ? (
                  <span className="tabular-nums">Δ —</span>
                ) : (
                  <span
                    className={[
                      "tabular-nums font-semibold",
                      delta > 0
                        ? "text-emerald-600 dark:text-emerald-400"
                        : delta < 0
                          ? "text-red-600 dark:text-red-400"
                          : "text-neutral-500 dark:text-neutral-500",
                    ].join(" ")}
                    title="Change versus grid position"
                  >
                    {delta > 0
                      ? `▲ ${delta} vs grid`
                      : delta < 0
                        ? `▼ ${Math.abs(delta)} vs grid`
                        : "—"}
                  </span>
                )}
              </div>
            </Card>
          </div>
        );
      })}
    </div>
  );
}