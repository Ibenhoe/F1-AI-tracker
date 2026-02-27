// src/components/prerace/TireStrategyPanel.jsx
import { useEffect, useMemo, useState } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";

import Card from "../ui/Card";
import TireBadge from "../ui/TireBadge";

function clamp(n, a, b) {
  const x = Number(n);
  if (!Number.isFinite(x)) return a;
  return Math.max(a, Math.min(b, x));
}

function toPct(n) {
  const x = Number(n);
  return Number.isFinite(x) ? clamp(Math.round(x), 0, 100) : 0;
}

function formatStrategyType(raw) {
  const s = String(raw || "").replace(/_/g, " ").trim();
  // keep it short (your screenshot shows SHORT labels like "MEDIUM STOP")
  return s ? s.toUpperCase() : "STRATEGY";
}

function formatPitStopsShort(laps) {
  if (!Array.isArray(laps) || laps.length === 0) return "Pit —";

  const first = Number(laps[0]);
  if (!Number.isFinite(first)) return "Pit —";

  if (laps.length === 1) return `Pit at Lap ${first}`;
  return `Pit at Lap ${first} +${laps.length - 1}`;
}

function isPrimaryRank(rank) {
  return Number(rank) === 1;
}

export default function TireStrategyPanel({ tireStrategies, circuitAnalysis }) {
  const list = Array.isArray(tireStrategies) ? tireStrategies : [];

  const sorted = useMemo(() => {
    return [...list].sort(
      (a, b) => Number(a.rank ?? 999) - Number(b.rank ?? 999)
    );
  }, [list]);

  const visibleCount = 3;
  const [start, setStart] = useState(0);
  const maxStart = Math.max(0, sorted.length - visibleCount);

  useEffect(() => {
    setStart((s) => Math.min(s, maxStart));
  }, [maxStart]);

  const canPrev = start > 0;
  const canNext = start < maxStart;

  const visible = sorted.slice(start, start + visibleCount);

  if (sorted.length === 0) {
    return (
      <Card className="p-6 text-center" bordered clip>
        <div className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">
          No strategy yet
        </div>
        <div className="mt-1 text-xs text-neutral-600 dark:text-neutral-400">
          Strategies will appear once the model finishes computing.
        </div>
      </Card>
    );
  }

  const wearPct =
    circuitAnalysis && circuitAnalysis.tire_wear_rate != null
      ? clamp(Number(circuitAnalysis.tire_wear_rate) * 100, 0, 100)
      : null;

  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      {/* Chevron row (same as PredictionsPanel) */}
      <div className="flex items-center justify-end px-1">
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => canPrev && setStart((s) => s - 1)}
            disabled={!canPrev}
            className={[
              "inline-flex h-8 w-8 items-center justify-center rounded-xl",
              "ring-1 ring-black/5 dark:ring-white/10",
              "bg-transparent hover:bg-black/[0.03] dark:hover:bg-white/[0.05]",
              "text-neutral-700 dark:text-neutral-200 transition-colors",
              "disabled:opacity-40 disabled:hover:bg-transparent",
            ].join(" ")}
            aria-label="Previous strategies"
          >
            <ChevronLeft size={16} />
          </button>

          <button
            type="button"
            onClick={() => canNext && setStart((s) => s + 1)}
            disabled={!canNext}
            className={[
              "inline-flex h-8 w-8 items-center justify-center rounded-xl",
              "border border-neutral-200/70 bg-white/60 backdrop-blur",
              "text-neutral-700 transition hover:bg-neutral-100/70",
              "disabled:opacity-40 disabled:hover:bg-white/60",
              "dark:border-white/10 dark:bg-neutral-950/30 dark:text-neutral-200 dark:hover:bg-white/5",
            ].join(" ")}
            aria-label="Next strategies"
          >
            <ChevronRight size={16} />
          </button>
        </div>
      </div>

      {/* 3-card grid (same as PredictionsPanel) */}
      <div className="grid min-h-0 flex-1 grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {visible.map((strat, localIdx) => {
          const idx = start + localIdx;

          const rank = Number(strat.rank ?? idx + 1);
          const primary = isPrimaryRank(rank);

          const confidence = toPct(strat.confidence ?? 0);
          const type = formatStrategyType(strat.strategy_type);

          const seq = Array.isArray(strat.tire_sequence) ? strat.tire_sequence : [];
          const pitShort = formatPitStopsShort(strat.pit_stop_laps);

          const recRaw =
            typeof strat.recommendation === "string" ? strat.recommendation.trim() : "";
          const rec = recRaw.length > 0 ? recRaw : null;

          return (
            <Card key={`${rank}-${idx}`} className="relative p-5" clip>
              {/* Top accent line */}
              <div
                className="absolute left-0 top-0 h-1 w-full"
                style={{
                  background: primary ? "rgb(var(--accent))" : "rgba(0,0,0,0.12)",
                }}
                aria-hidden="true"
              />

              {/* Header anatomy: keep tight + no overflow */}
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="text-[11px] font-semibold uppercase tracking-widest text-neutral-500 dark:text-neutral-400">
                    Tire strategy
                  </div>

                  {/* Title: allow 2 lines max, never overflow */}
                  <div className="mt-2 truncate text-lg font-semibold tracking-tight text-neutral-900 dark:text-neutral-50">
                    {type}
                  </div>
                </div>

                {/* Pill: same styling; ALT shows rank */}
                <div
                  className={[
                    "shrink-0 rounded-full px-3 py-1.5",
                    "text-sm font-semibold tabular-nums",
                    "bg-neutral-100 text-neutral-900 ring-1 ring-black/10",
                    "dark:bg-white/10 dark:text-neutral-50 dark:ring-white/10",
                  ].join(" ")}
                  title={primary ? "Primary strategy" : `Rank ${rank}`}
                >
                  {primary ? "PRIMARY" : `ALT ${rank}`}
                </div>
              </div>

              {/* Confidence row (same pattern as Predictions) */}
              <div className="mt-4 flex items-baseline justify-between gap-3">
                <div className="text-sm text-neutral-600 dark:text-neutral-400">
                  Confidence
                </div>
                <div className="text-base font-semibold tabular-nums text-neutral-900 dark:text-neutral-50">
                  {confidence}%
                </div>
              </div>

              <div className="mt-3">
                <div className="h-2 w-full overflow-hidden rounded-full bg-neutral-200/70 dark:bg-white/10">
                  <div
                    className="h-full rounded-full"
                    style={{ width: `${confidence}%`, background: "rgb(var(--accent))" }}
                    aria-hidden="true"
                  />
                </div>
              </div>

              {/* Tire sequence: keep in ONE line, never wrap */}
              <div className="mt-4 flex items-center gap-2 overflow-hidden">
                {seq.length > 0 ? (
                  <div className="flex items-center gap-2 overflow-hidden">
                    {seq.slice(0, 3).map((compound, i) => (
                      <div key={`${compound}-${i}`} className="flex items-center gap-2">
                        <TireBadge compound={compound} />
                        {i < Math.min(seq.length, 3) - 1 ? (
                          <span className="text-xs text-neutral-400">→</span>
                        ) : null}
                      </div>
                    ))}

                    {/* If sequence is longer, keep it readable */}
                    {seq.length > 3 ? (
                      <span className="text-xs text-neutral-500 dark:text-neutral-400 tabular-nums">
                        +{seq.length - 3}
                      </span>
                    ) : null}
                  </div>
                ) : (
                  <span className="text-xs text-neutral-500 dark:text-neutral-400">
                    No sequence
                  </span>
                )}
              </div>

              {/* Bottom meta row: short + single line (like PredictionsPanel) */}
              <div className="mt-4 flex items-center justify-between text-[11px] text-neutral-600 dark:text-neutral-400">
                <span className="tabular-nums truncate">{pitShort}</span>
                <span className="tabular-nums">
                  Degradation  {wearPct == null ? "—" : `${Math.round(wearPct)}%`}
                </span>
              </div>
            </Card>
          );
        })}

        {/* same padding behavior as PredictionsPanel */}
        {visible.length < 3
          ? Array.from({ length: 3 - visible.length }).map((_, i) => (
            <div key={`pad-${i}`} className="hidden lg:block" />
          ))
          : null}

        {visible.length < 2 ? <div className="hidden sm:block lg:hidden" /> : null}
      </div>
    </div>
  );
}