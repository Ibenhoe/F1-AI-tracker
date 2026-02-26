import { ArrowDown, ArrowUp, Minus } from "lucide-react";
import TireBadge from "../components/ui/TireBadge";
import { getTeamColor } from "../utils/teamColors";

function PosDelta({ value }) {
  const v = Number(value ?? 0);

  if (v > 0) {
    return (
      <span className="inline-flex items-center justify-end gap-1 text-xs font-semibold text-emerald-600 dark:text-emerald-400 tabular-nums">
        <ArrowUp size={14} />+{v}
      </span>
    );
  }

  if (v < 0) {
    return (
      <span className="inline-flex items-center justify-end gap-1 text-xs font-semibold text-red-600 dark:text-red-400 tabular-nums">
        <ArrowDown size={14} />
        {v}
      </span>
    );
  }

  return (
    <span className="inline-flex items-center justify-end gap-1 text-xs font-semibold text-neutral-500 dark:text-neutral-400 tabular-nums">
      <Minus size={14} />0
    </span>
  );
}

function formatGap(raw, isLeader) {
  if (isLeader) return "Leader";
  if (raw == null) return "—";

  const str = String(raw).trim();
  if (!str || str === "—" || str.toLowerCase() === "nan") return "—";

  // Accept inputs like "+1.2", "1.234", "1.2s", "+1.2s"
  const cleaned = str.replace(/s$/i, "").replace("+", "");
  const num = Number.parseFloat(cleaned);

  if (!Number.isFinite(num)) return str; // fallback to whatever backend gives

  // If 0 for non-leader, still show 0.0s (but keep it subtle)
  const sign = num > 0 ? "+" : "";
  return `${sign}${num.toFixed(1)}s`;
}

export default function DriversList({ drivers }) {
  const list = Array.isArray(drivers) ? drivers : [];

  return (
    <div className="overflow-hidden rounded-2xl bg-white/70 ring-1 ring-neutral-200/70 backdrop-blur-sm dark:bg-[rgb(var(--panel))] dark:ring-white/10 dark:backdrop-blur-none">
      {/* Header row */}
      <div
        className={[
          "grid min-w-0",
          "grid-cols-[44px_1fr_92px_64px_72px_60px] md:grid-cols-[44px_1fr_110px_92px_64px_72px_60px]",
          "items-center gap-3 px-4 py-2.5",
          "text-[11px] font-semibold uppercase tracking-widest",
          "text-neutral-500 dark:text-neutral-400",
          "border-b border-neutral-200/70 dark:border-white/10",
          "bg-transparent",
        ].join(" ")}
      >
        <div className="text-center">Pos</div>
        <div className="min-w-0">Driver</div>
        <div className="hidden text-right md:block">Last lap</div>
        <div className="text-right">Tires</div>
        <div className="text-right">Pits</div>
        <div className="text-right">Gap</div>
        <div className="text-right">Δ</div>
      </div>

      {/* Rows */}
      <div className="divide-y divide-neutral-200/70 dark:divide-white/10">
        {list.length > 0 ? (
          list.map((driver, idx) => {
            const name = driver.driver_name || driver.driver_code || "Unknown";
            const team = driver.team || "—";
            const teamColor = getTeamColor(team);

            const pos =
              Number.isFinite(Number(driver.position)) && Number(driver.position) > 0
                ? Number(driver.position)
                : idx + 1;

            const isLeader = pos === 1 || idx === 0;

            const gapDisplay = formatGap(driver.gap, isLeader);

            const lapTime = driver.lap_time && driver.lap_time !== "—" ? driver.lap_time : "—";

            return (
              <div
                key={driver.driver_id ?? driver.driver_code ?? `${idx}-${name}`}
                className={[
                  "relative",
                  "grid min-w-0",
                  "grid-cols-[44px_1fr_92px_64px_72px_60px] md:grid-cols-[44px_1fr_110px_92px_64px_72px_60px]",
                  "items-center gap-3 px-4 py-3",
                  "bg-white/60 hover:bg-neutral-50/80",
                  "dark:bg-transparent dark:hover:bg-white/[0.04]",
                  "transition-colors",
                ].join(" ")}
              >
                {/* Team hairline accent */}
                <div
                  className="absolute left-0 top-0 h-full w-[2px]"
                  style={{
                    backgroundColor: teamColor ? teamColor : "rgba(0,0,0,0.08)",
                    opacity: teamColor ? 0.75 : 1,
                  }}
                  aria-hidden="true"
                />

                {/* Position */}
                <div className="text-center text-sm font-semibold tabular-nums text-neutral-900 dark:text-neutral-100">
                  {pos}
                </div>

                {/* Driver */}
                <div className="min-w-0">
                  <div className="truncate text-sm font-semibold text-neutral-900 dark:text-neutral-100">
                    {name}
                  </div>
                  <div className="truncate text-xs text-neutral-500 dark:text-neutral-500">
                    {team}
                  </div>

                  {/* On mobile, show last lap under the driver for better density */}
                  <div className="mt-1 flex items-center justify-between gap-2 md:hidden">
                    <span className="text-[11px] font-medium text-neutral-400 dark:text-neutral-600">
                      Last lap
                    </span>
                    <span className="text-xs font-semibold tabular-nums text-neutral-700 dark:text-neutral-300">
                      {lapTime}
                    </span>
                  </div>
                </div>

                {/* Last lap (desktop) */}
                <div className="hidden text-right md:block">
                  <div className="text-sm font-semibold tabular-nums text-neutral-900 dark:text-neutral-100">
                    {lapTime}
                  </div>
                </div>

                {/* Tires */}
                <div className="flex items-center justify-end gap-2">
                  <TireBadge compound={driver.tire_compound} size={24} />
                  <span className="text-sm font-semibold tabular-nums text-neutral-900 dark:text-neutral-100">
                    {driver.tire_age ?? 0}
                  </span>
                </div>

                {/* Pits */}
                <div className="text-right text-sm font-semibold tabular-nums text-neutral-900 dark:text-neutral-100">
                  {driver.pit_stops ?? 0}
                </div>

                {/* Gap */}
                <div className="text-right text-sm font-semibold tabular-nums">
                  <span
                    className={
                      isLeader
                        ? "text-amber-600 dark:text-amber-300"
                        : gapDisplay === "—"
                          ? "text-neutral-400 dark:text-neutral-600"
                          : "text-neutral-700 dark:text-neutral-300"
                    }
                  >
                    {gapDisplay}
                  </span>
                </div>

                {/* Δ Pos */}
                <div className="text-right">
                  <PosDelta value={driver.position_change} />
                </div>
              </div>
            );
          })
        ) : (
          <div className="px-6 py-12 text-center bg-neutral-50/60 dark:bg-transparent">
            <div className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">
              Waiting for race data…
            </div>
            <div className="mt-1 text-xs text-neutral-500 dark:text-neutral-500">
              Standings will appear once telemetry is available.
            </div>
          </div>
        )}
      </div>
    </div>
  );
}