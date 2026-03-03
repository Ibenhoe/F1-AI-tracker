// src/components/dashboard/components/StatsBar.jsx
import Card from "../../ui/Card";
import { getTeamColor } from "../../../utils/teamColors";

export default function StatsBar({ drivers = [], currentLap = 0, totalLaps = 0, trackStatus = "1" }) {
  const sorted = [...drivers].sort(
    (a, b) => (a?.position ?? 999) - (b?.position ?? 999)
  );

  const leader = sorted.length > 0 ? sorted[0] : null;

  const fastestDriver = sorted.reduce((best, d) => {
    if (!d?.lap_time || d.lap_time === "—") return best;
    if (!best) return d;
    return d.lap_time < best.lap_time ? d : best;
  }, null);

  const pct =
    totalLaps > 0 ? Math.min(100, Math.round((currentLap / totalLaps) * 100)) : 0;

  const FLAG_MAP = {
    "1": { label: "Green Flag", sub: "Track clear", color: "#10B981" },
    "2": { label: "Yellow Flag", sub: "Hazard on track", color: "#FACC15" },
    "4": { label: "Safety Car", sub: "SC deployed", color: "#F97316" },
    "5": { label: "Red Flag", sub: "Race suspended", color: "#EF4444" },
    "6": { label: "Virtual SC", sub: "VSC deployed", color: "#F59E0B" },
  };

  const flag = FLAG_MAP[String(trackStatus)] || FLAG_MAP["1"];
  const leaderColor = leader ? getTeamColor(leader.team) : null;

  const Tile = ({ title, accentColor, children }) => (
    <Card className="relative overflow-hidden p-5" clip>
      {accentColor ? (
        <div
          className="absolute left-0 top-0 h-1 w-full opacity-80"
          style={{ background: accentColor }}
        />
      ) : null}

      <p className="text-[11px] font-semibold uppercase tracking-widest text-neutral-500 dark:text-neutral-500">
        {title}
      </p>

      <div className="mt-3">{children}</div>
    </Card>
  );

  return (
    <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
      {/* Race Progress */}
      <Tile title="Race Progress" accentColor="rgb(var(--accent))">
        <div className="flex items-baseline gap-2">
          <p className="text-2xl font-semibold tabular-nums text-neutral-900 dark:text-neutral-50">
            {currentLap}
          </p>
          <p className="text-sm text-neutral-500 dark:text-neutral-400">
            / {totalLaps} laps
          </p>
        </div>

        <div className="mt-4">
          <div className="h-2 w-full overflow-hidden rounded-full bg-neutral-200/70 dark:bg-white/10">
            <div
              className="h-full rounded-full transition-all duration-700"
              style={{
                width: `${pct}%`,
                background: "rgb(var(--accent))",
              }}
            />
          </div>
        </div>
      </Tile>

      {/* Race Leader */}
      <Tile title="Race Leader" accentColor={leaderColor}>
        {leader ? (
          <>
            <p
              className="text-2xl font-semibold tracking-tight"
              style={{ color: leaderColor || "inherit" }}
            >
              {leader.driver_code}
            </p>
            <p className="mt-1 text-sm text-neutral-500 dark:text-neutral-400">
              {leader.team}
            </p>
          </>
        ) : (
          <p className="text-sm text-neutral-400">—</p>
        )}
      </Tile>

      {/* Fastest Lap */}
      <Tile title="Fastest Lap" accentColor="#A855F7">
        {fastestDriver ? (
          <>
            <p className="text-2xl font-semibold text-purple-500 dark:text-purple-400">
              {fastestDriver.driver_code}
            </p>
            <p className="mt-1 text-sm font-mono text-purple-400">
              {fastestDriver.lap_time}
            </p>
          </>
        ) : (
          <p className="text-sm text-neutral-400">—</p>
        )}
      </Tile>

      {/* Race Status */}
      <Tile title="Race Status" accentColor={flag.color}>
        <div className="flex items-center justify-between">
          <div>
            <p className="text-lg font-semibold" style={{ color: flag.color }}>
              {flag.label}
            </p>
            <p className="mt-1 text-sm text-neutral-500 dark:text-neutral-400">
              {flag.sub}
            </p>
          </div>

          <div
            className="h-3 w-3 rounded-full"
            style={{ background: flag.color }}
          />
        </div>
      </Tile>
    </div>
  );
}