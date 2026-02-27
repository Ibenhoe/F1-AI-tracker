// src/components/prerace/PreRaceStatTiles.jsx
import Card from "../ui/Card";
import { getTeamColor } from "../../utils/teamColors";

function clamp(n, a, b) {
  const x = Number(n);
  if (!Number.isFinite(x)) return a;
  return Math.max(a, Math.min(b, x));
}

/* EXACT tile style as Dashboard StatsBar */
function StatTile({ title, accentColor, children }) {
  return (
    <Card
      className={[
        "relative overflow-hidden p-5",
        // Force same height as Dashboard tiles
        "h-[152px]",          // <-- als het nét niet matcht: 148/150/156 proberen
        "flex flex-col",
      ].join(" ")}
      clip
    >
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

      {/* Make inner content fill and distribute vertically */}
      <div className="mt-3 flex-1 min-h-0 flex flex-col justify-between">
        {children}
      </div>
    </Card>
  );
}

function formatDriverName(x) {
  const name = x?.driver_name || x?.driver || "—";
  return String(name);
}

export default function PreRaceStatTiles({ movers, circuitAnalysis }) {
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
    <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
      {/* Biggest Riser */}
      <StatTile title="Biggest Riser" accentColor={riserColor}>
        <div>
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
        </div>

        <p
          className={[
            "text-sm font-semibold tabular-nums",
            riserDelta > 0
              ? "text-emerald-600 dark:text-emerald-400"
              : "text-neutral-400 dark:text-neutral-600",
          ].join(" ")}
        >
          {riserDelta > 0 ? `▲ ${riserDelta}` : "—"}
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
            "mt-2 text-sm font-semibold tabular-nums",
            fallerDelta < 0
              ? "text-red-600 dark:text-red-400"
              : "text-neutral-400 dark:text-neutral-600",
          ].join(" ")}
        >
          {fallerDelta < 0 ? `▼ ${Math.abs(fallerDelta)}` : "—"}
        </p>
      </StatTile>

      {/* Highest Confidence */}
      <StatTile title="Highest Confidence" accentColor={confColor}>
        <div>
          <div className="flex items-baseline justify-between gap-3">
            <p
              className="min-w-0 truncate text-2xl font-semibold tracking-tight"
              style={{ color: confColor || "inherit" }}
              title={formatDriverName(topConf)}
            >
              {formatDriverName(topConf)}
            </p>

            <p className="shrink-0 text-sm font-semibold tabular-nums" style={{ color: confColor }}>
              {Math.round(conf)}%
            </p>
          </div>

          <p className="mt-1 text-sm text-neutral-500 dark:text-neutral-400">
            {topConf?.team || "—"}
          </p>
        </div>

        <div className="h-2 w-full overflow-hidden rounded-full bg-neutral-200/70 dark:bg-white/10">
          <div
            className="h-full rounded-full transition-all duration-700"
            style={{ width: `${conf}%`, background: confColor }}
            aria-hidden="true"
          />
        </div>
      </StatTile>
    </div>
  );
}