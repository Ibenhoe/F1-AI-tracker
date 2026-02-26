import { getTeamColor } from "../utils/teamColors";

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

function battleTone(gapSeconds) {
  const hot = gapSeconds < 0.3;

  return {
    hot,
    label: hot ? "ATTACK" : "DRS",
    pill: hot
      ? "bg-red-500/10 text-red-700 dark:bg-red-500/10 dark:text-red-200"
      : "bg-amber-400/15 text-amber-800 dark:bg-amber-400/10 dark:text-amber-200",
    bar: hot ? "bg-red-500" : "bg-amber-400",
  };
}

function GapBar({ gap }) {
  // gap: seconds between cars (0..1)
  const pct = clamp((1 - gap) * 100, 0, 100);
  const tone = battleTone(gap);

  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-neutral-200/70 dark:bg-white/10">
      <div className={`h-full rounded-full ${tone.bar}`} style={{ width: `${pct}%` }} />
    </div>
  );
}

export default function BattlesWidget({ drivers = [] }) {
  if (!drivers || drivers.length < 2) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-sm text-neutral-500 dark:text-neutral-400">
          Waiting for race data…
        </p>
      </div>
    );
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
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-sm text-neutral-500 dark:text-neutral-400">
          No DRS battles at the moment.
        </p>
      </div>
    );
  }

  return (
    <div className="h-full min-h-0 overflow-auto">
      {/* Same surface language as Notifications */}
      <div className="divide-y divide-neutral-200/70 rounded-2xl bg-white/70 ring-1 ring-neutral-200/70 backdrop-blur-sm dark:divide-white/10 dark:bg-[rgb(var(--panel))] dark:ring-white/10 dark:backdrop-blur-none">
        {battles.map((b) => {
          const tone = battleTone(b.between);

          const behindCode = b.behind.driver_code || "—";
          const aheadCode = b.ahead.driver_code || "—";

          const behindTeam = b.behind.team || "";
          const aheadTeam = b.ahead.team || "";

          // Subtle team hint (no hardcoded mapping; pulled from utils)
          const railColor = behindTeam ? getTeamColor(behindTeam) : null;

          return (
            <div key={`${b.ahead.driver_code}-${b.behind.driver_code}`} className="relative px-4 py-3">
              {/* Optional ultra-subtle rail (can remove if you want zero color hints) */}
              {railColor ? (
                <div
                  className="absolute left-0 top-0 h-full w-[3px] opacity-70"
                  style={{ backgroundColor: railColor }}
                  aria-hidden="true"
                />
              ) : null}

              {/* Top row */}
              <div className="flex items-center gap-3">
                <div className="min-w-[108px]">
                  <div className="text-sm font-medium tabular-nums text-neutral-900 dark:text-neutral-50">
                    P{b.behind.position} <span className="tracking-wide">{behindCode}</span>
                  </div>
                </div>

                <div className="min-w-0 flex-1">
                  <GapBar gap={b.between} />
                </div>

                <div
                  className={[
                    "shrink-0 rounded-full px-2.5 py-1",
                    "text-xs font-semibold tabular-nums",
                    tone.pill,
                  ].join(" ")}
                  title="Gap to car ahead"
                >
                  {tone.label} +{b.between.toFixed(3)}s
                </div>
              </div>

              {/* Subline */}
              <div className="mt-2 text-xs text-neutral-500 dark:text-neutral-400">
                Chasing{" "}
                <span className="font-medium text-neutral-900 dark:text-neutral-200">
                  P{b.ahead.position} {aheadCode}
                </span>
                {aheadTeam ? (
                  <span className="text-neutral-400 dark:text-neutral-500"> ({aheadTeam})</span>
                ) : null}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}