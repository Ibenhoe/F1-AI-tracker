import { useEffect, useMemo, useState } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import Card from "./ui/Card";
import { getTeamColor } from "../utils/teamColors";

function clamp01(n) {
  const x = Number(n);
  if (!Number.isFinite(x)) return 0;
  return Math.min(100, Math.max(0, x));
}

function toInt(n, fallback = 0) {
  const x = Number(n);
  return Number.isFinite(x) ? Math.round(x) : fallback;
}

const DRIVER_TO_TEAM = {
  VER: "Red Bull Racing",
  PER: "Red Bull Racing",
  LEC: "Ferrari",
  SAI: "Ferrari",
  HAM: "Mercedes",
  RUS: "Mercedes",
  NOR: "McLaren",
  PIA: "McLaren",
  ALO: "Aston Martin",
  STR: "Aston Martin",
  GAS: "Alpine",
  OCO: "Alpine",
  ALB: "Williams",
  SAR: "Williams",
  BOT: "Kick Sauber",
  ZHO: "Kick Sauber",
  HUL: "Haas F1 Team",
  MAG: "Haas F1 Team",
  TSU: "RB",
  RIC: "RB",
};

function pickTeam(pred) {
  if (!pred) return "";
  if (typeof pred.team === "string" && pred.team) return pred.team;
  if (typeof pred.team_name === "string" && pred.team_name) return pred.team_name;
  if (typeof pred.constructor_name === "string" && pred.constructor_name) return pred.constructor_name;
  const code = String(pred.driver_code || "").trim().toUpperCase();
  return DRIVER_TO_TEAM[code] || "";
}

export default function PredictionsPanel({ predictions }) {
  const list = Array.isArray(predictions) ? predictions : [];
  const top = useMemo(() => list.slice(0, 5), [list]);

  const visibleCount = 3;
  const [start, setStart] = useState(0);
  const maxStart = Math.max(0, top.length - visibleCount);

  useEffect(() => {
    setStart((s) => Math.min(s, maxStart));
  }, [maxStart]);

  const canPrev = start > 0;
  const canNext = start < maxStart;

  const visible = top.slice(start, start + visibleCount);

  if (top.length === 0) {
    return (
      <Card className="p-6 text-center" bordered clip>
        <div className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">
          Model is training
        </div>
        <div className="mt-1 text-xs text-neutral-600 dark:text-neutral-400">
          Predictions appear once enough laps are processed.
        </div>
      </Card>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
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
            aria-label="Previous predictions"
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
            aria-label="Next predictions"
          >
            <ChevronRight size={16} />
          </button>
        </div>
      </div>

      <div className="grid min-h-0 flex-1 grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {visible.map((pred, localIdx) => {
          const idx = start + localIdx;

          const driver = pred.driver_name || pred.driver_code || "Unknown";
          const team = pickTeam(pred);
          const teamColor = getTeamColor(team) || "rgba(0,0,0,0.12)";

          const fromPos = toInt(pred.position ?? 0);
          const toPos = toInt(pred.prediction ?? 0);
          const confidence = clamp01(pred.confidence ?? 0);

          return (
            <Card
              key={`${pred.driver_code ?? driver}-${idx}`}
              className="relative p-5"
              clip
            >
              <div className="absolute left-0 top-0 h-1 w-full" style={{ background: teamColor }} />

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

                <div
                  className={[
                    "shrink-0 rounded-full px-3 py-1.5",
                    "text-sm font-semibold tabular-nums",
                    "bg-neutral-100 text-neutral-900 ring-1 ring-black/10",
                    "dark:bg-white/10 dark:text-neutral-50 dark:ring-white/10",
                  ].join(" ")}
                >
                  P{idx + 1}
                </div>
              </div>

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
                  />
                </div>
              </div>

              <div className="mt-4 flex items-center justify-between text-[11px] text-neutral-600 dark:text-neutral-400">
                <span className="tabular-nums">
                  Pos {fromPos} → {toPos}
                </span>
                <span className="tabular-nums">
                  Pit {toInt(pred.pit_stops ?? 0)}
                </span>
              </div>
            </Card>
          );
        })}

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