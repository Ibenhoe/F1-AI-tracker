// src/components/prerace/PreRaceStatTiles.jsx
import Card from "../ui/Card";
import { getTeamColor } from "../../utils/teamColors";

function clamp(n, a, b) {
  const x = Number(n);
  if (!Number.isFinite(x)) return a;
  return Math.max(a, Math.min(b, x));
}

/* Same tile style as Dashboard StatsBar */
function StatTile({ title, accentColor, children }) {
  return (
    <Card className="relative overflow-hidden p-5" clip>
      {accentColor ? (
        <div
          className="absolute left-0 top-0 h-1 w-full opacity-80"
          style={{ background: accentColor }}
          aria-hidden="true"
        />
      ) : null}

      <p className="text-[11px] font-semibold uppercase tracking-widest text-neutral-500 dark:text-neutral-500">
        {title}
      </p>

      <div className="mt-3">{children}</div>
    </Card>
  );
}

function formatDriverName(x) {
  const name = x?.driver_name || x?.driver || "—";
  return String(name);
}

export default function PreRaceStatTiles({ movers }) {
  const riser = movers?.riser;
  const faller = movers?.faller;
  const topConf = movers?.topConf;

  const riserDelta = Number(riser?.delta ?? 0);
  const fallerDelta = Number(faller?.delta ?? 0);
  const conf = clamp(Number(topConf?.confidence ?? 0), 0, 100);

  const riserColor = getTeamColor(riser?.team) || "rgb(var(--accent))";
  const fallerColor = getTeamColor(faller?.team) || "rgb(var(--accent))";
  const confColor = getTeamColor(topConf?.team) || "rgb(var(--accent))";

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
      {/* Biggest Riser */}
      <StatTile title="Biggest Riser" accentColor={riserColor}>
        <p
          className="truncate text-2xl font-semibold tracking-tight"
          style={{ color: riserColor || "inherit" }}
          title={formatDriverName(riser)}
        >
          {formatDriverName(riser)}
        </p>

        <p className="mt-1 text-sm text-neutral-500 dark:text-neutral-400">
          {riser?.team || "—"}
        </p>

        <p
          className={[
            "mt-3 text-sm font-semibold tabular-nums",
            riserDelta > 0
              ? "text-emerald-600 dark:text-emerald-400"
              : "text-neutral-400 dark:text-neutral-600",
          ].join(" ")}
        >
          {riserDelta > 0 ? `▲ ${riserDelta} positions` : "No change"}
        </p>
      </StatTile>

      {/* Biggest Faller */}
      <StatTile title="Biggest Faller" accentColor={fallerColor}>
        <p
          className="truncate text-2xl font-semibold tracking-tight"
          style={{ color: fallerColor || "inherit" }}
          title={formatDriverName(faller)}
        >
          {formatDriverName(faller)}
        </p>

        <p className="mt-1 text-sm text-neutral-500 dark:text-neutral-400">
          {faller?.team || "—"}
        </p>

        <p
          className={[
            "mt-3 text-sm font-semibold tabular-nums",
            fallerDelta < 0
              ? "text-red-600 dark:text-red-400"
              : "text-neutral-400 dark:text-neutral-600",
          ].join(" ")}
        >
          {fallerDelta < 0 ? `▼ ${Math.abs(fallerDelta)} positions` : "No change"}
        </p>
      </StatTile>

      {/* Highest Confidence */}
      <StatTile title="Highest Confidence" accentColor={confColor}>
        <p
          className="truncate text-2xl font-semibold tracking-tight"
          style={{ color: confColor || "inherit" }}
          title={formatDriverName(topConf)}
        >
          {formatDriverName(topConf)}
        </p>

        <p className="mt-1 text-sm text-neutral-500 dark:text-neutral-400">
          {topConf?.team || "—"}
        </p>

        <div className="mt-3 flex items-center justify-between gap-4">
          <p
            className="text-lg font-semibold tabular-nums"
            style={{ color: confColor || "rgb(var(--accent))" }}
          >
            {Math.round(conf)}%
          </p>

          <div className="h-2 w-full max-w-[160px] overflow-hidden rounded-full bg-black/[0.06] dark:bg-white/[0.08]">
            <div
              className="h-2 rounded-full"
              style={{ width: `${conf}%`, background: confColor }}
              aria-hidden="true"
            />
          </div>
        </div>
      </StatTile>
    </div>
  );
}