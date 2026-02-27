import Card from "../ui/Card";
import Badge from "../ui/Badge";
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
  return s ? s.toUpperCase() : "STRATEGY";
}

function formatPitStops(laps) {
  if (!Array.isArray(laps) || laps.length === 0) return "None";
  return laps.map((lap) => `Lap ${lap}`).join(", ");
}

function ConfidenceBar({ value }) {
  const pct = toPct(value);

  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-black/5 dark:bg-white/10">
      <div
        className="h-full rounded-full bg-[rgb(var(--accent))]"
        style={{ width: `${pct}%` }}
        aria-hidden="true"
      />
    </div>
  );
}

function RankPill({ rank }) {
  const isPrimary = Number(rank) === 1;

  return (
    <span
      className={[
        "shrink-0 rounded-full px-2.5 py-1",
        "text-xs font-semibold tabular-nums",
        isPrimary
          ? "bg-[rgb(var(--accent))] text-[rgb(var(--accent-fg))] ring-1 ring-black/5 dark:ring-white/10"
          : "bg-black/[0.04] text-neutral-800 ring-1 ring-black/5 dark:bg-white/[0.05] dark:text-neutral-200 dark:ring-white/10",
      ].join(" ")}
    >
      {isPrimary ? "PRIMARY" : "ALT"}
    </span>
  );
}

export default function TireStrategyPanel({ tireStrategies, circuitAnalysis }) {
  const list = Array.isArray(tireStrategies) ? tireStrategies : [];

  // Empty state (keep it simple + consistent)
  if (list.length === 0) {
    return (
      <div className="flex h-full min-h-[220px] items-center justify-center">
        <p className="text-sm text-neutral-500 dark:text-neutral-400">
          No strategy available yet.
        </p>
      </div>
    );
  }

  const wearPct = circuitAnalysis
    ? clamp(Number(circuitAnalysis.tire_wear_rate ?? 0) * 100, 0, 100)
    : null;

  return (
    <div className="h-full min-h-0 overflow-auto">
      {/* Battles-style container: one card with dividers */}
      <Card className="divide-y divide-black/5 dark:divide-white/10" clip bordered>
        {list.map((strat, idx) => {
          const rank = Number(strat.rank ?? idx + 1);
          const isPrimary = rank === 1;

          const confidence = toPct(strat.confidence ?? 0);
          const type = formatStrategyType(strat.strategy_type);

          const pitStops = formatPitStops(strat.pit_stop_laps);
          const seq = Array.isArray(strat.tire_sequence) ? strat.tire_sequence : [];
          const rec = typeof strat.recommendation === "string" ? strat.recommendation.trim() : "";

          return (
            <div
              key={`${rank}-${idx}`}
              className="relative px-4 py-3 transition-colors hover:bg-black/[0.02] dark:hover:bg-white/[0.03]"
            >
              {/* Subtle left rail for primary only (Battles vibe) */}
              {isPrimary ? (
                <div
                  className="absolute left-px top-0 h-full w-[3px] opacity-70"
                  style={{ backgroundColor: "rgb(var(--accent))" }}
                  aria-hidden="true"
                />
              ) : null}

              {/* Top row: pill + type + confidence */}
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <RankPill rank={rank} />

                    <span className="truncate text-[11px] font-semibold uppercase tracking-widest text-neutral-500 dark:text-neutral-400">
                      {type}
                    </span>
                  </div>

                  <div className="mt-2 text-xs text-neutral-500 dark:text-neutral-400">
                    Pit stops:{" "}
                    <span className="font-medium text-neutral-900 dark:text-neutral-200">
                      {pitStops}
                    </span>
                  </div>
                </div>

                <div className="shrink-0 text-right">
                  <div className="tabular-nums text-lg font-bold text-neutral-900 dark:text-neutral-100">
                    {confidence}%
                  </div>
                  <div className="text-[11px] text-neutral-500 dark:text-neutral-400">
                    confidence
                  </div>
                </div>
              </div>

              {/* Confidence bar (like Battles gap bar) */}
              <div className="mt-3">
                <ConfidenceBar value={confidence} />
              </div>

              {/* Tire sequence */}
              {seq.length > 0 ? (
                <div className="mt-3 flex flex-wrap items-center gap-1.5">
                  {seq.map((compound, i) => (
                    <div key={`${compound}-${i}`} className="flex items-center gap-1.5">
                      <TireBadge compound={compound} />
                      {i < seq.length - 1 ? (
                        <span className="text-xs text-neutral-400">→</span>
                      ) : null}
                    </div>
                  ))}
                </div>
              ) : null}

              {/* Recommendation (compact, Battles-like subline) */}
              {rec ? (
                <div className="mt-3 text-xs leading-relaxed text-neutral-600 dark:text-neutral-400">
                  {rec}
                </div>
              ) : null}
            </div>
          );
        })}
      </Card>
    </div>
  );
}