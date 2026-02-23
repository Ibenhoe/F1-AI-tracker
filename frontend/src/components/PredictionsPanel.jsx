import { Brain } from "lucide-react";

function clamp01(n) {
  const x = Number(n);
  if (!Number.isFinite(x)) return 0;
  return Math.min(100, Math.max(0, x));
}

export default function PredictionsPanel({ predictions, currentLap, modelMetrics }) {
  const list = Array.isArray(predictions) ? predictions : [];
  const top = list.slice(0, 5);

  const maturity = Math.round(Number(modelMetrics?.model_maturity_percentage ?? 0));
  const updates = Number(modelMetrics?.total_updates ?? 0);

  const mae =
    modelMetrics && modelMetrics.recent_mae_average !== undefined
      ? Number(modelMetrics.recent_mae_average)
      : null;

  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      {/* context */}
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2 text-xs text-neutral-600 dark:text-neutral-400">
            <Brain size={14} className="text-[rgb(var(--accent))] opacity-90" />
            <span>Expected top-5 finish order (AI)</span>
          </div>
        </div>
      </div>

      {/* model meta */}
      <div className="rounded-xl px-1">
        <div className="flex items-center justify-between gap-3 px-2">
          <div className="flex items-baseline gap-2">
            <span className="text-[10px] font-semibold uppercase tracking-widest text-neutral-500/80 dark:text-neutral-500">
              Updates
            </span>
            <span className="text-sm font-semibold tabular-nums text-neutral-900 dark:text-neutral-100">
              {updates}
            </span>
          </div>

          <div className="flex items-baseline gap-2">
            <span className="text-[10px] font-semibold uppercase tracking-widest text-neutral-500/80 dark:text-neutral-500">
              Maturity
            </span>
            <span className="text-sm font-semibold tabular-nums text-neutral-900 dark:text-neutral-100">
              {maturity}%
            </span>
          </div>
        </div>

        <div className="mt-2 px-2">
          <div className="h-1 w-full overflow-hidden rounded-full bg-neutral-200/70 dark:bg-white/10">
            <div
              className="h-full rounded-full bg-[rgb(var(--accent))]"
              style={{ width: `${clamp01(maturity)}%` }}
            />
          </div>
        </div>

        {mae !== null ? (
          <div className="mt-2 flex items-center justify-between px-2 text-[11px] text-neutral-600 dark:text-neutral-400">
            <span className="tabular-nums">MAE {mae.toFixed(2)}</span>
            <span className="opacity-70">Lower is better</span>
          </div>
        ) : null}
      </div>

      {/* Predictions */}
      {top.length === 0 ? (
        <div className="rounded-2xl bg-white ring-1 ring-neutral-200/70 px-4 py-10 text-center dark:bg-white/5 dark:ring-white/10">
          <div className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">
            Model is training
          </div>
          <div className="mt-1 text-xs text-neutral-600 dark:text-neutral-400">
            Predictions appear once enough laps are processed.
          </div>
        </div>
      ) : (
        <div
          className={[
            "flex-1 min-h-0",
            "flex items-start gap-4 overflow-x-auto", // ✅ was items-stretch
            "pb-2",
            "snap-x snap-mandatory",
            "[scrollbar-width:thin]",
          ].join(" ")}
        >
          {top.map((pred, idx) => {
            const driver = pred.driver_name || pred.driver_code || "Unknown";

            const team =
              typeof pred.team === "string"
                ? pred.team
                : typeof pred.team_name === "string"
                  ? pred.team_name
                  : typeof pred.constructor_name === "string"
                    ? pred.constructor_name
                    : "";

            const fromPos = Math.round(Number(pred.position ?? 0));
            const toPos = Math.round(Number(pred.prediction ?? 0));
            const confidence = clamp01(pred.confidence ?? 0);

            const podium = `P${idx + 1}`;

            return (
              <div
                key={`${pred.driver_code ?? driver}-${idx}`}
                className="snap-start min-w-[280px] max-w-[340px] flex-1 h-auto relative" // ✅ was h-full
              >
                {/* P badge */}
                <div className="absolute left-1/2 top-4 z-10 -translate-x-1/2">
                  <div
                    className={[
                      "rounded-full px-4 py-1.5",
                      "text-sm font-semibold tabular-nums",
                      "bg-[rgb(var(--accent))]",
                      "text-neutral-950 dark:text-[rgb(var(--accent-fg))]",
                      "ring-1 ring-black/10 dark:ring-white/10",
                      "shadow-[0_12px_28px_rgba(0,0,0,0.25)] dark:shadow-[0_12px_28px_rgba(0,0,0,0.45)]",
                    ].join(" ")}
                  >
                    {podium}
                  </div>
                </div>

                {/* Card */}
                <div
                  className={[
                    "h-auto flex flex-col", // ✅ was h-full
                    "rounded-2xl px-6 pt-12 pb-5",
                    "overflow-hidden",
                    // LIGHT MODE
                    "bg-white ring-1 ring-neutral-200/70",
                    "shadow-[0_1px_0_rgba(0,0,0,0.04),0_18px_50px_rgba(0,0,0,0.10)]",
                    // DARK MODE
                    "dark:bg-neutral-950/30 dark:ring-white/10",
                    "dark:shadow-[0_1px_0_rgba(255,255,255,0.04),0_18px_50px_rgba(0,0,0,0.55)]",
                    "text-center",
                  ].join(" ")}
                >
                  <div className="min-w-0">
                    {team ? (
                      <div className="text-[11px] font-semibold uppercase tracking-widest text-neutral-500 dark:text-neutral-400">
                        {team}
                      </div>
                    ) : (
                      <div className="h-[14px]" />
                    )}

                    <div className="mt-2 truncate text-xl font-semibold tracking-tight text-neutral-900 dark:text-neutral-50">
                      {driver}
                    </div>

                    <div className="mt-2 text-sm text-neutral-700 dark:text-neutral-300">
                      <span className="font-semibold tabular-nums text-neutral-900 dark:text-neutral-50">
                        {Math.round(confidence)}%
                      </span>{" "}
                      confidence
                    </div>

                    <div className="mt-4 flex justify-center gap-2 text-[11px] text-neutral-600 dark:text-neutral-400">
                      <span className="tabular-nums">Pos {fromPos} → {toPos}</span>
                      <span className="text-neutral-300 dark:text-neutral-700">•</span>
                      <span className="tabular-nums">Pit {pred.pit_stops ?? 0}</span>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}