const TEAM_COLORS = {
  Mercedes: "#00D7B6",
  "Red Bull Racing": "#4781D7",
  Ferrari: "#ED1131",
  McLaren: "#F47600",
  Alpine: "#00A1E8",
  "Racing Bulls": "#6C98FF",
  "Aston Martin": "#229971",
  Williams: "#1868DB",
  "Kick Sauber": "#01C00E",
  Haas: "#9C9FA2",
};

function clamp(n, a, b) {
  const x = Number(n);
  if (!Number.isFinite(x)) return a;
  return Math.min(b, Math.max(a, x));
}

function parseGap(gapStr) {
  if (!gapStr) return Infinity;
  const s = String(gapStr).replace("+", "").replace("s", "").trim();
  const n = parseFloat(s);
  return Number.isFinite(n) ? n : Infinity;
}

function teamRail(team) {
  const c = TEAM_COLORS[team] || "#8a8a8a";
  return (
    <div
      className="absolute left-0 top-0 h-full w-1.5"
      style={{ backgroundColor: c }}
      aria-hidden="true"
    />
  );
}

function battleTone(gapSeconds) {
  const hot = gapSeconds < 0.3;

  return {
    hot,
    label: hot ? "ATTACK" : "DRS",
    pill: hot
      ? "bg-red-500/10 text-red-700 ring-1 ring-red-500/20 dark:bg-red-500/10 dark:text-red-200 dark:ring-red-500/25"
      : "bg-amber-400/15 text-amber-800 ring-1 ring-amber-400/25 dark:bg-amber-400/10 dark:text-amber-200 dark:ring-amber-400/25",
    bar: hot ? "bg-red-500" : "bg-amber-400",
  };
}

function GapBar({ gap }) {
  const pct = clamp((1 - gap) * 100, 0, 100);
  const tone = battleTone(gap);

  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-neutral-200/80 dark:bg-white/10">
      <div className={`h-full rounded-full ${tone.bar}`} style={{ width: `${pct}%` }} />
    </div>
  );
}

export default function BattlesWidget({ drivers = [] }) {
  if (!drivers || drivers.length < 2) {
    return <p className="text-xs text-neutral-600 dark:text-neutral-400">Waiting for race data…</p>;
  }

  const sorted = [...drivers].sort((a, b) => a.position - b.position);

  const battles = [];
  for (let i = 0; i < sorted.length - 1; i++) {
    const ahead = sorted[i];
    const behind = sorted[i + 1];

    const aheadGap = parseGap(ahead.gap);
    const behindGap = parseGap(behind.gap);
    if (!Number.isFinite(aheadGap) || !Number.isFinite(behindGap)) continue;

    const between = behind.position === 1 ? 0 : behindGap - aheadGap;
    if (between >= 0 && between < 1.0) battles.push({ ahead, behind, between });
  }

  if (battles.length === 0) {
    return <p className="text-xs text-neutral-600 dark:text-neutral-400">No DRS battles at the moment.</p>;
  }

  return (
    <div className="space-y-3">
      {battles.map((b) => {
        const tone = battleTone(b.between);

        const behindCode = b.behind.driver_code || "—";
        const aheadCode = b.ahead.driver_code || "—";

        const behindTeam = b.behind.team || "";
        const aheadTeam = b.ahead.team || "";

        return (
          <div
            key={`${b.ahead.driver_code}-${b.behind.driver_code}`}
            className={[
              "relative overflow-hidden",
              "rounded-2xl px-4 py-3 pl-5", // space for rail
              // LIGHT MODE
              "bg-white ring-1 ring-neutral-200/70",
              "shadow-[0_1px_0_rgba(0,0,0,0.04),0_14px_40px_rgba(0,0,0,0.10)]",
              // DARK MODE
              "dark:bg-neutral-950/25 dark:ring-white/10",
              "dark:shadow-[0_1px_0_rgba(255,255,255,0.04),0_18px_50px_rgba(0,0,0,0.45)]",
            ].join(" ")}
          >
            {teamRail(behindTeam)}

            {/* Top row */}
            <div className="flex items-center gap-3">
              <div className="min-w-[110px]">
                <div className="text-sm font-semibold tabular-nums text-neutral-900 dark:text-neutral-50">
                  P{b.behind.position} <span className="tracking-wide">{behindCode}</span>
                </div>
              </div>

              <div className="min-w-0 flex-1">
                <GapBar gap={b.between} />
              </div>

              <div
                className={[
                  "shrink-0 rounded-full px-3 py-1",
                  "text-xs font-semibold tabular-nums",
                  tone.pill,
                ].join(" ")}
                title="Gap to car ahead"
              >
                {tone.label} +{b.between.toFixed(3)}s
              </div>
            </div>

            {/* Subline */}
            <div className="mt-2 text-xs text-neutral-600 dark:text-neutral-400">
              Chasing{" "}
              <span className="font-medium text-neutral-900 dark:text-neutral-200">
                P{b.ahead.position} {aheadCode}
              </span>
              {aheadTeam ? (
                <span className="text-neutral-500 dark:text-neutral-500"> ({aheadTeam})</span>
              ) : null}
            </div>
          </div>
        );
      })}
    </div>
  );
}