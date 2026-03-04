import React, { useMemo, useState } from "react";
import Badge from "./ui/Badge.jsx";
import { getTeamColor } from "../utils/teamColors";

// Reuse RaceReplay SegmentedControl (same as RightPanel)
import SegmentedControl from "./racereplay/ui/controls/SegmentedControl";

// Fallback team map by driver code (works even when backend returns "Unknown")
const DRIVER_TEAMS = {
  VER: "Red Bull Racing",
  PER: "Red Bull Racing",
  HAM: "Mercedes",
  RUS: "Mercedes",
  LEC: "Ferrari",
  SAI: "Ferrari",
  NOR: "McLaren",
  PIA: "McLaren",
  ALO: "Aston Martin",
  STR: "Aston Martin",
  GAS: "Alpine",
  OCO: "Alpine",
  TSU: "Racing Bulls",
  RIC: "Racing Bulls",
  LAW: "Racing Bulls",
  ALB: "Williams",
  SAR: "Williams",
  COL: "Williams",
  HUL: "Haas",
  MAG: "Haas",
  BEA: "Haas",
  BOT: "Kick Sauber",
  ZHO: "Kick Sauber",
};

const DRIVER_LAST = {
  VER: "Verstappen",
  LEC: "Leclerc",
  SAI: "Sainz",
  PIA: "Piastri",
  NOR: "Norris",
  HAM: "Hamilton",
  RUS: "Russell",
  ALO: "Alonso",
  STR: "Stroll",
  GAS: "Gasly",
  OCO: "Ocon",
  MAG: "Magnussen",
  HUL: "Hülkenberg",
  BOT: "Bottas",
  ZHO: "Zhou",
  TSU: "Tsunoda",
  ALB: "Albon",
  SAR: "Sargeant",
  PER: "Pérez",
  RIC: "Ricciardo",
  LAW: "Lawson",
  BEA: "Bearman",
  COL: "Colapinto",
};

function tireMeta(compound) {
  const c = String(compound ?? "").toUpperCase();
  if (c.startsWith("S"))
    return {
      label: "S",
      bg: "bg-red-500/15",
      text: "text-red-600 dark:text-red-400",
      ring: "ring-red-500/35",
    };
  if (c.startsWith("M"))
    return {
      label: "M",
      bg: "bg-amber-500/15",
      text: "text-amber-700 dark:text-amber-400",
      ring: "ring-amber-500/35",
    };
  if (c.startsWith("H"))
    return {
      label: "H",
      bg: "bg-neutral-400/20",
      text: "text-neutral-700 dark:text-neutral-300",
      ring: "ring-neutral-400/35",
    };
  if (c.startsWith("I") || c.startsWith("INT"))
    return {
      label: "I",
      bg: "bg-emerald-500/15",
      text: "text-emerald-700 dark:text-emerald-400",
      ring: "ring-emerald-500/35",
    };
  if (c.startsWith("W"))
    return {
      label: "W",
      bg: "bg-blue-500/15",
      text: "text-blue-700 dark:text-blue-400",
      ring: "ring-blue-500/35",
    };
  return {
    label: "?",
    bg: "bg-black/[0.03] dark:bg-white/[0.06]",
    text: "text-neutral-600 dark:text-neutral-300",
    ring: "ring-black/10 dark:ring-white/10",
  };
}

function formatGap(gap) {
  if (gap == null) return "—";
  const num = parseFloat(String(gap).replace("+", ""));
  if (Number.isNaN(num)) return "—";
  if (num >= 60) {
    const m = Math.floor(num / 60);
    const s = (num % 60).toFixed(1).padStart(4, "0");
    return `+${m}:${s}`;
  }
  return `+${num.toFixed(3)}`;
}

function withAlpha(color, alpha) {
  const c = String(color || "").trim();

  // hex: #RRGGBB
  if (/^#([0-9a-f]{6})$/i.test(c)) {
    const a = Math.round(Math.max(0, Math.min(1, alpha)) * 255)
      .toString(16)
      .padStart(2, "0");
    return `${c}${a}`;
  }

  // rgb()/rgba()
  const rgbMatch = c.match(/^rgb\(\s*([^)]+)\s*\)$/i);
  if (rgbMatch) return `rgba(${rgbMatch[1]}, ${alpha})`;

  const rgbaMatch = c.match(/^rgba\(\s*([^)]+)\s*\)$/i);
  if (rgbaMatch) return `rgba(${rgbaMatch[1].split(",").slice(0, 3).join(",")}, ${alpha})`;

  // fallback: just return original (better than breaking)
  return c;
}

/**
 * ReplayLeaderboard – themed standings list (no external CSS)
 */
export default function ReplayLeaderboard({
  drivers,
  selectedDriver,
  onDriverSelect,
  currentLap,
  totalLaps,
}) {
  const [gapMode, setGapMode] = useState("leader"); // 'leader' | 'interval'

  const sorted = useMemo(() => {
    return Object.entries(drivers || {})
      .map(([code, d]) => ({ code, ...(d || {}) }))
      .sort((a, b) => (a.position ?? 999) - (b.position ?? 999));
  }, [drivers]);

  const withInterval = useMemo(() => {
    return sorted.map((d, i) => {
      if (i === 0) return { ...d, interval: null };
      const myGap = parseFloat(String(d.gap ?? "").replace("+", "")) || 0;
      const prevGap = parseFloat(String(sorted[i - 1].gap ?? "").replace("+", "")) || 0;
      const iv = Math.max(0, myGap - prevGap);
      return { ...d, interval: `+${iv.toFixed(3)}` };
    });
  }, [sorted]);

  const running = sorted.filter((d) => d.status !== "OUT").length;
  const dnfCount = sorted.length - running;

  const gapItems = useMemo(
    () => [
      { id: "leader", label: "To leader" },
      { id: "interval", label: "Interval" },
    ],
    []
  );

  return (
    <div className="flex h-full min-h-0 flex-col gap-4">
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-[11px] font-semibold uppercase tracking-widest text-neutral-400 dark:text-neutral-600">
            Standings
          </p>
          <p className="mt-1 text-sm font-semibold tracking-tight text-neutral-900 dark:text-neutral-50">
            {running} running{dnfCount > 0 ? ` · ${dnfCount} DNF` : ""}
          </p>
        </div>

        {currentLap != null ? (
          <Badge variant="neutral">
            Lap {currentLap}
            {totalLaps ? `/${totalLaps}` : ""}
          </Badge>
        ) : null}
      </div>

      {/* Gap mode toggle (theme-consistent) */}
      <SegmentedControl
        value={gapMode}
        onChange={setGapMode}
        ariaLabel="Gap mode"
        items={gapItems}
      />

      {/* List container (theme) */}
      <div
        className={[
          "min-h-0 flex-1 overflow-hidden",
          "rounded-2xl",
          "bg-white dark:bg-neutral-950/40",
          "ring-1 ring-black/5 dark:ring-white/10",
        ].join(" ")}
      >
        {/* Column headers */}
        <div
          className={[
            "sticky top-0 z-10",
            "bg-white/80 dark:bg-neutral-950/60 backdrop-blur",
            "border-b border-black/5 dark:border-white/10",
            "grid grid-cols-[30px_1fr_44px_64px] items-center gap-2",
            "px-4 py-2.5",
            "text-[11px] font-semibold uppercase tracking-widest",
            "text-neutral-500 dark:text-neutral-400",
          ].join(" ")}
        >
          <div className="text-center">P</div>
          <div>Driver</div>
          <div className="text-center">Tyre</div>
          <div className="text-right">{gapMode === "leader" ? "Gap" : "Int."}</div>
        </div>

        <div className="min-h-0 h-full overflow-y-auto">
          <div className="divide-y divide-black/5 dark:divide-white/10">
            {withInterval.length === 0 ? (
              <div className="px-4 py-10 text-center text-sm text-neutral-500 dark:text-neutral-400">
                Waiting for data…
              </div>
            ) : (
              withInterval.map((driver, i) => {
                const isSelected = selectedDriver === driver.code;
                const isLeader = i === 0;
                const isDNF = driver.status === "OUT";

                const teamRaw =
                  driver.team && driver.team !== "Unknown"
                    ? driver.team
                    : DRIVER_TEAMS[driver.code] || "";

                const team = teamRaw; // keep original for now (data)
                const teamColor = getTeamColor(teamRaw) || "rgb(var(--accent))";
                const lastName = DRIVER_LAST[driver.code] || "";

                const tire = tireMeta(driver.tire_compound);

                const gapStr =
                  gapMode === "leader"
                    ? formatGap(driver.gap)
                    : isLeader
                      ? null
                      : formatGap(driver.interval);

                return (
                  <button
                    key={driver.code}
                    type="button"
                    onClick={() => onDriverSelect(isSelected ? null : driver.code)}
                    className={[
                      "relative w-full text-left",
                      "grid grid-cols-[30px_1fr_44px_64px] items-center gap-2",
                      "px-4 py-2.5",
                      "transition-colors",
                      "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[rgb(var(--accent))]",
                      isSelected
                        ? "bg-black/[0.03] dark:bg-white/[0.06]"
                        : "hover:bg-black/[0.02] dark:hover:bg-white/[0.03]",
                      isDNF ? "opacity-45" : "",
                    ].join(" ")}
                  >
                    {/* Accent rail */}
                    <div
                      className="absolute left-0 top-0 h-full w-1 opacity-90"
                      style={{ backgroundColor: isSelected ? teamColor : withAlpha(teamColor, 0.6) }}
                      aria-hidden="true"
                    />

                    {/* Position */}
                    <div
                      className={[
                        "text-center text-xs font-black tabular-nums",
                        isLeader && !isDNF
                          ? "text-amber-600 dark:text-amber-400"
                          : "text-neutral-700 dark:text-neutral-300",
                      ].join(" ")}
                    >
                      {isDNF ? (
                        <span className="text-[10px] font-black text-red-500">
                          OUT
                        </span>
                      ) : (
                        Math.round(driver.position ?? 0)
                      )}
                    </div>

                    {/* Driver */}
                    <div className="min-w-0">
                      <div className="flex items-baseline gap-2 min-w-0">
                        <span className="text-sm font-semibold tracking-tight text-neutral-900 dark:text-neutral-50">
                          {driver.code}
                        </span>
                        {lastName ? (
                          <span className="truncate text-xs text-neutral-500 dark:text-neutral-400">
                            {lastName}
                          </span>
                        ) : null}
                      </div>

                      <div className="mt-0.5 flex items-center gap-2 min-w-0">
                        {teamRaw ? (
                          <span className="truncate text-[11px] text-neutral-500 dark:text-neutral-500">
                            {String(teamRaw)
                              .replace(" Racing", "")
                              .replace(" F1 Team", "")
                              .replace("Racing Bulls", "RB")}
                          </span>
                        ) : null}

                        {driver.pit_stops > 0 ? (
                          <span className="text-[11px] text-neutral-400 dark:text-neutral-600">
                            • P{driver.pit_stops}
                          </span>
                        ) : null}
                      </div>
                    </div>

                    {/* Tyre */}
                    <div className="flex flex-col items-center gap-1">
                      <span
                        className={[
                          "inline-flex h-7 w-7 items-center justify-center rounded-full",
                          "text-[11px] font-black",
                          "ring-1",
                          tire.bg,
                          tire.text,
                          tire.ring,
                        ].join(" ")}
                        title={`${driver.tire_compound || "?"} – ${driver.tire_age ?? 0} laps old`}
                      >
                        {tire.label}
                      </span>

                      {driver.tire_age != null ? (
                        <span className="text-[10px] tabular-nums text-neutral-400 dark:text-neutral-600 leading-none">
                          {driver.tire_age}L
                        </span>
                      ) : null}
                    </div>

                    {/* Gap / Interval */}
                    <div className="text-right">
                      {isLeader ? (
                        <span className="text-[11px] font-black text-amber-600 dark:text-amber-400">
                          LEAD
                        </span>
                      ) : gapStr && gapStr !== "—" ? (
                        <span className="text-[12px] tabular-nums font-semibold text-neutral-700 dark:text-neutral-300">
                          {gapStr}
                        </span>
                      ) : (
                        <span className="text-[12px] text-neutral-400">—</span>
                      )}
                    </div>
                  </button>
                );
              })
            )}
          </div>
        </div>
      </div>
    </div>
  );
}