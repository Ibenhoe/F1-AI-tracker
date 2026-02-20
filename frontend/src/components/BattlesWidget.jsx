/**
 * BattlesWidget – shows current DRS-range battles (gap < 1.0 s).
 * Also displays the fastest-lap holder with a purple accent (like F1 TV).
 */

const TEAM_COLORS = {
  "Mercedes":         "#00D7B6",
  "Red Bull Racing":  "#4781D7",
  "Ferrari":          "#ED1131",
  "McLaren":          "#F47600",
  "Alpine":           "#00A1E8",
  "Racing Bulls":     "#6C98FF",
  "Aston Martin":     "#229971",
  "Williams":         "#1868DB",
  "Kick Sauber":      "#01C00E",
  "Haas":             "#9C9FA2",
};

function teamDot(team) {
  const c = TEAM_COLORS[team] || "#888";
  return (
    <span
      className="inline-block h-2 w-2 rounded-full shrink-0"
      style={{ background: c }}
    />
  );
}

function parseGap(gapStr) {
  if (!gapStr) return Infinity;
  const s = String(gapStr).replace("+", "").replace("s", "").trim();
  const n = parseFloat(s);
  return isNaN(n) ? Infinity : n;
}

function GapMeter({ gap }) {
  // 0 s → full bar (hot), 1 s → empty bar
  const pct = Math.max(0, Math.min(100, ((1 - gap) / 1) * 100));
  const hot = gap < 0.3;
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-neutral-200 dark:bg-neutral-700">
      <div
        className={`h-full rounded-full transition-all duration-500 ${hot ? "bg-red-500" : "bg-amber-400"}`}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

export default function BattlesWidget({ drivers = [] }) {
  if (!drivers || drivers.length < 2) {
    return (
      <p className="text-xs text-neutral-500 dark:text-neutral-400">
        Waiting for race data…
      </p>
    );
  }

  // Sort by position
  const sorted = [...drivers].sort((a, b) => a.position - b.position);

  // Find all battles (< 1.0 s gap, excluding P1 gap "+0.000")
  const battles = [];
  for (let i = 0; i < sorted.length - 1; i++) {
    const ahead = sorted[i];
    const behind = sorted[i + 1];
    const gap = parseGap(behind.gap) - parseGap(ahead.gap);
    // behind.gap minus ahead.gap gives the gap between them
    // simpler: just parse behind.gap if ahead is P1 etc.
    // Actually the gap field is cumulative gap to P1.
    // So gap between consecutive = behind.gap - ahead.gap
    const aheadGap = parseGap(ahead.gap);
    const behindGap = parseGap(behind.gap);
    const between = behind.position === 1 ? 0 : behindGap - aheadGap;
    if (between >= 0 && between < 1.0) {
      battles.push({ ahead, behind, between });
    }
  }

  // Fastest lap: the driver with the lowest lap_time (if we track it)
  // We don't have explicit fastest lap field, so we skip that here.
  // We'll show it separately in the parent if needed.

  if (battles.length === 0) {
    return (
      <p className="text-xs text-neutral-500 dark:text-neutral-400">
        No DRS battles at the moment.
      </p>
    );
  }

  return (
    <div className="space-y-2">
      {battles.map((b) => {
        const hot = b.between < 0.3;
        return (
          <div
            key={`${b.ahead.driver_code}-${b.behind.driver_code}`}
            className={[
              "rounded-xl border p-3 transition-colors",
              hot
                ? "border-red-400/40 bg-red-50/60 dark:border-red-500/20 dark:bg-red-500/5"
                : "border-amber-300/40 bg-amber-50/60 dark:border-amber-500/20 dark:bg-amber-500/5",
            ].join(" ")}
          >
            <div className="flex items-center justify-between gap-2 mb-2">
              <div className="flex items-center gap-1.5 text-sm font-medium text-neutral-900 dark:text-neutral-100">
                {teamDot(b.behind.team)}
                <span>P{b.behind.position} {b.behind.driver_code}</span>
              </div>
              <div
                className={`text-xs font-semibold tabular-nums px-2 py-0.5 rounded-full ${
                  hot
                    ? "bg-red-500/15 text-red-700 dark:text-red-300"
                    : "bg-amber-400/15 text-amber-800 dark:text-amber-300"
                }`}
              >
                {hot ? "ATTACK" : "DRS"}
                &nbsp;+{b.between.toFixed(3)}s
              </div>
            </div>

            <GapMeter gap={b.between} />

            <p className="mt-1.5 text-xs text-neutral-500 dark:text-neutral-400">
              Chasing&nbsp;
              <span className="font-medium text-neutral-700 dark:text-neutral-300">
                P{b.ahead.position} {b.ahead.driver_code}
              </span>
              &nbsp;({b.ahead.team})
            </p>
          </div>
        );
      })}
    </div>
  );
}
