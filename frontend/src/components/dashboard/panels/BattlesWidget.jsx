// src/components/dashboard/panels/BattlesWidget.jsx
import { getTeamColor } from "../../../utils/teamColors";
import Card from "../../ui/Card";

function clamp(n, a, b) {
  const x = Number(n);
  if (!Number.isFinite(x)) return a;
  return Math.min(b, Math.max(a, x));
}

function parseGap(gapStr) {
  if (gapStr == null) return Infinity;

  const s = String(gapStr).trim();
  if (!s || s === "—") return Infinity;
  if (s.toLowerCase() === "leader") return 0;

  const cleaned = s.replace("+", "").replace(/s$/i, "");
  const n = Number.parseFloat(cleaned);
  return Number.isFinite(n) ? n : Infinity;
}

function battleTone(gapSeconds) {
  const hot = gapSeconds < 0.3;

  return {
    hot,
    label: hot ? "ATTACK" : "DRS",
    pill: hot
      ? "bg-black/[0.04] text-red-700 ring-1 ring-black/5 dark:bg-white/[0.05] dark:text-red-200 dark:ring-white/10"
      : "bg-black/[0.04] text-amber-800 ring-1 ring-black/5 dark:bg-white/[0.05] dark:text-amber-200 dark:ring-white/10",
    bar: hot ? "bg-red-500/80" : "bg-amber-400/80",
  };
}

function GapBar({ gap }) {
  // gap: seconds between cars (0..1)
  const pct = clamp((1 - gap) * 100, 0, 100);
  const tone = battleTone(gap);

  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-black/5 dark:bg-white/10">
      <div className={`h-full rounded-full ${tone.bar}`} style={{ width: `${pct}%` }} />
    </div>
  );
}

function toPos(v, fallback = 9999) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

export default function BattlesWidget({ drivers = [] }) {
  const list = Array.isArray(drivers) ? drivers : [];

  if (list.length < 2) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-sm text-neutral-500 dark:text-neutral-400">
          Waiting for race data…
        </p>
      </div>
    );
  }

  const sorted = [...list].sort((a, b) => toPos(a.position) - toPos(b.position));

  const battles = [];
  for (let i = 0; i < sorted.length - 1; i++) {
    const ahead = sorted[i];
    const behind = sorted[i + 1];

    const aheadGap = parseGap(ahead?.gap);
    const behindGap = parseGap(behind?.gap);

    if (!Number.isFinite(aheadGap) || !Number.isFinite(behindGap)) continue;

    const between = behindGap - aheadGap;

    // Only show close battles (0..1s)
    if (between >= 0 && between < 1.0) {
      battles.push({ ahead, behind, between });
    }
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
      <Card className="divide-y divide-black/5 dark:divide-white/10" clip bordered>
        {battles.map((b, idx) => {
          const tone = battleTone(b.between);

          const behindCode = b.behind?.driver_code || "—";
          const aheadCode = b.ahead?.driver_code || "—";

          const behindTeam = b.behind?.team || "";
          const aheadTeam = b.ahead?.team || "";

          const railColor = behindTeam ? getTeamColor(behindTeam) : null;

          const key =
            (b.ahead?.driver_id && b.behind?.driver_id)
              ? `${b.ahead.driver_id}-${b.behind.driver_id}`
              : `${aheadCode}-${behindCode}-${idx}`;

          return (
            <div
              key={key}
              className="relative px-4 py-3 transition-colors hover:bg-black/[0.02] dark:hover:bg-white/[0.03]"
            >
              {railColor ? (
                <div
                  className="absolute left-px top-0 h-full w-[3px] opacity-70"
                  style={{ backgroundColor: railColor }}
                  aria-hidden="true"
                />
              ) : null}

              <div className="flex items-center gap-3">
                <div className="min-w-[108px]">
                  <div className="text-sm font-medium tabular-nums text-neutral-900 dark:text-neutral-50">
                    P{toPos(b.behind?.position, 0)}{" "}
                    <span className="tracking-wide">{behindCode}</span>
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

              <div className="mt-2 text-xs text-neutral-500 dark:text-neutral-400">
                Chasing{" "}
                <span className="font-medium text-neutral-900 dark:text-neutral-200">
                  P{toPos(b.ahead?.position, 0)} {aheadCode}
                </span>
                {aheadTeam ? (
                  <span className="text-neutral-400 dark:text-neutral-500">
                    {" "}
                    ({aheadTeam})
                  </span>
                ) : null}
              </div>
            </div>
          );
        })}
      </Card>
    </div>
  );
}