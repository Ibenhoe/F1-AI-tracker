import {
  ArrowDownRight,
  ArrowRight,
  ArrowUpRight,
  Brain,
  Activity,
} from "lucide-react";
import Badge from "./ui/Badge.jsx";

function modelStatus(metrics) {
  if (!metrics) {
    return {
      label: "Initializing",
      variant: "warning",
    };
  }

  const maturity = Number(metrics.model_maturity_percentage ?? 0);

  if (maturity >= 100) return { label: "Optimized", variant: "success" };
  if (maturity >= 50) return { label: "Training", variant: "warning" };
  if (maturity >= 25) return { label: "Learning", variant: "neutral" };
  return { label: "Initializing", variant: "warning" };
}

function TrendIcon({ trend }) {
  if (trend === "up") return <ArrowUpRight size={16} className="text-emerald-300" />;
  if (trend === "down") return <ArrowDownRight size={16} className="text-red-300" />;
  return <ArrowRight size={16} className="text-neutral-400" />;
}

export default function PredictionsPanel({
  predictions,
  currentLap,
  modelMetrics,
}) {
  const list = Array.isArray(predictions) ? predictions : [];
  const top = list.slice(0, 5);

  const status = modelStatus(modelMetrics);
  const maturity = Math.round(Number(modelMetrics?.model_maturity_percentage ?? 0));
  const updates = Number(modelMetrics?.total_updates ?? 0);

  const hasComponents =
    modelMetrics?.sgd_model_ready ||
    modelMetrics?.mlp_model_ready ||
    modelMetrics?.rf_classifier_ready;

  const mae =
    modelMetrics && modelMetrics.recent_mae_average !== undefined
      ? Number(modelMetrics.recent_mae_average)
      : null;

  const maeTrend = String(modelMetrics?.mae_trend ?? "").toLowerCase();

  const maeTrendLabel =
    maeTrend === "improving" ? "Improving" : maeTrend ? "Stable" : null;

  return (
    <div className="space-y-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-semibold tracking-tight">Predictions</h2>
            <Brain size={16} className="text-neutral-500" />
          </div>
          <p className="mt-1 text-xs text-neutral-600 dark:text-neutral-400">
            Top 5 expected finishing order (AI)
          </p>
        </div>

        <div className="flex flex-wrap items-center justify-end gap-2">
          <Badge variant="neutral">Lap {currentLap ?? 0}</Badge>
          <Badge variant={status.variant}>{status.label}</Badge>
        </div>
      </div>

      {/* Model meta */}
      <div className="rounded-xl border border-neutral-200 bg-neutral-50 p-3 dark:border-neutral-800 dark:bg-neutral-950/40">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex flex-wrap items-center gap-2 text-xs text-neutral-600 dark:text-neutral-400">
            <span className="inline-flex items-center gap-2">
              <Activity size={14} className="text-neutral-500" />
              <span>
                Updates:{" "}
                <span className="font-medium text-neutral-900 dark:text-neutral-200 tabular-nums">
                  {updates}
                </span>
              </span>
            </span>

            <span className="text-neutral-400 dark:text-neutral-600">•</span>

            <span>
              Maturity:{" "}
              <span className="font-medium text-neutral-900 dark:text-neutral-200 tabular-nums">
                {maturity}%
              </span>
            </span>
          </div>

          {hasComponents ? (
            <div className="flex flex-wrap items-center gap-2">
              {modelMetrics?.sgd_model_ready ? (
                <Badge variant="neutral">SGD</Badge>
              ) : null}
              {modelMetrics?.mlp_model_ready ? (
                <Badge variant="neutral">MLP</Badge>
              ) : null}
              {modelMetrics?.rf_classifier_ready ? (
                <Badge variant="neutral">RF</Badge>
              ) : null}
            </div>
          ) : (
            <div className="text-xs text-neutral-600 dark:text-neutral-500">
              Components warming up…
            </div>
          )}
        </div>


        <div className="mt-3 h-2 w-full rounded-full bg-neutral-200 dark:bg-neutral-900">
          <div
            className="h-2 rounded-full bg-neutral-900 dark:bg-neutral-100"
            style={{ width: `${Math.min(100, Math.max(0, maturity))}%` }}
          />
        </div>
      </div>

      {/* Predictions list */}
      {top.length === 0 ? (
        <div className="rounded-xl border border-neutral-200 bg-neutral-50 px-4 py-10 text-center dark:border-neutral-800 dark:bg-neutral-950/40">
          <div className="text-sm font-medium text-neutral-900 dark:text-neutral-200">
            Model is training…
          </div>
          <div className="mt-1 text-xs text-neutral-600 dark:text-neutral-500">
            Predictions will appear once enough laps have been processed.
          </div>
        </div>
      ) : (
        <div className="space-y-2">
          {top.map((pred, idx) => {
            const driver = pred.driver_name || pred.driver_code || "Unknown";
            const fromPos = Math.round(Number(pred.position ?? 0));
            const toPos = Math.round(Number(pred.prediction ?? 0));
            const confidence = Math.round(Number(pred.confidence ?? 0));

            return (
              <div
                key={`${pred.driver_code ?? driver}-${idx}`}
                className="rounded-xl border border-neutral-200 bg-white px-3 py-3 dark:border-neutral-800 dark:bg-neutral-950/40"
              >
                <div className="flex items-start gap-3">
                  <div className="w-12 shrink-0">
                    <div className="text-xs text-neutral-600 dark:text-neutral-500">Rank</div>
                    <div className="mt-1 flex items-center gap-2">
                      <span className="text-sm font-semibold text-neutral-900 dark:text-neutral-100 tabular-nums">
                        #{idx + 1}
                      </span>
                      <TrendIcon trend={pred.trend} />
                    </div>
                  </div>

                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="truncate text-sm font-medium text-neutral-900 dark:text-neutral-100">
                        {driver}
                      </div>
                      <div className="text-xs text-neutral-600 dark:text-neutral-400 tabular-nums">
                        Pos {fromPos} → {toPos}
                      </div>
                    </div>

                    <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-xs text-neutral-600 dark:text-neutral-500">
                      <span className="tabular-nums">Start P{pred.grid_pos ?? "—"}</span>
                      <span className="text-neutral-400 dark:text-neutral-700">•</span>
                      <span className="tabular-nums">Pit stops {pred.pit_stops ?? 0}</span>
                    </div>

                    <div className="mt-3 flex items-center gap-3">
                      <div className="h-2 flex-1 rounded-full bg-neutral-200 dark:bg-neutral-900">
                        <div
                          className="h-2 rounded-full bg-neutral-900 dark:bg-neutral-100"
                          style={{
                            width: `${Math.min(100, Math.max(0, confidence))}%`,
                          }}
                        />
                      </div>
                      <div className="w-12 text-right text-xs font-medium text-neutral-700 dark:text-neutral-300 tabular-nums">
                        {confidence}%
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Footer: performance */}
      {mae !== null ? (
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-neutral-200 bg-neutral-50 px-3 py-3 dark:border-neutral-800 dark:bg-neutral-950/40">
          <div className="text-xs text-neutral-600 dark:text-neutral-500">Model performance</div>
          <div className="flex flex-wrap items-center gap-3 text-xs text-neutral-700 dark:text-neutral-300">
            <span className="tabular-nums">MAE {mae.toFixed(2)}</span>
            {maeTrendLabel ? (
              <span className="text-neutral-500 dark:text-neutral-500">{maeTrendLabel}</span>
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  );
}
