// src/components/wiki/components/WikiTable.jsx
import Card from "../../ui/Card";
import { getTeamColor } from "../../../utils/teamColors";

function EmptyState({ title, subtitle }) {
  return (
    <div className="px-6 py-12 text-center">
      <div className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">
        {title}
      </div>
      {subtitle ? (
        <div className="mt-1 text-xs text-neutral-500 dark:text-neutral-500">
          {subtitle}
        </div>
      ) : null}
    </div>
  );
}

export default function WikiTable({
  viewType,
  loading,
  tableData,
  formatName,
  onDriverClick,
}) {
  const list = Array.isArray(tableData) ? tableData : [];

  const headClass =
    viewType === "qualifying"
      ? "grid-cols-[44px_1fr_96px_96px_96px]"
      : "grid-cols-[44px_1fr_140px_64px]";

  return (
    <Card className="overflow-hidden" clip bordered>
      {/* Header row (DriversList style) */}
      <div
        className={[
          "grid min-w-0",
          headClass,
          "items-center gap-3 px-4 py-2.5",
          "text-[11px] font-semibold uppercase tracking-widest",
          "text-neutral-500 dark:text-neutral-400",
          "border-b border-black/5 dark:border-white/10",
          "bg-transparent",
        ].join(" ")}
      >
        <div className="text-center">Pos</div>
        <div className="min-w-0">Driver</div>

        {viewType === "race" ? (
          <>
            <div className="text-right">Time/Status</div>
            <div className="text-right">Pts</div>
          </>
        ) : null}

        {viewType === "qualifying" ? (
          <>
            <div className="text-center">Q1</div>
            <div className="text-center">Q2</div>
            <div className="text-center">Q3</div>
          </>
        ) : null}
      </div>

      {/* Rows */}
      <div className="divide-y divide-black/5 dark:divide-white/10">
        {loading ? (
          <EmptyState title="Fetching data…" subtitle="Results will appear shortly." />
        ) : list.length === 0 ? (
          <EmptyState
            title="No data available"
            subtitle="Try a different race or session view."
          />
        ) : (
          list.map((row, idx) => {
            const pos =
              Number.isFinite(Number(row.position)) && Number(row.position) > 0
                ? Number(row.position)
                : idx + 1;

            const driverRaw = row.driver;
            const driver = formatName
              ? formatName(driverRaw)
              : String(driverRaw || "Unknown");

            const team = row.team || "—";
            const teamColor = getTeamColor(team);

            const time = row.time ?? "—";
            const pts = Number(row.points ?? 0);
            const ptsText = pts > 0 ? `+${pts}` : "—";

            const q1 = row.q1 ?? "—";
            const q2 = row.q2 ?? "—";
            const q3 = row.q3 ?? "—";

            return (
              <div
                key={`${pos}-${driver}-${team}-${idx}`}
                className={[
                  "relative",
                  "grid min-w-0",
                  headClass,
                  "items-center gap-3 px-4 py-3",
                  "bg-transparent hover:bg-black/[0.02] dark:hover:bg-white/[0.03]",
                  "transition-colors",
                ].join(" ")}
              >
                {/* Team hairline accent */}
                <div
                  className="absolute left-0 top-0 h-full w-[2px]"
                  style={{
                    backgroundColor: teamColor || "transparent",
                    opacity: teamColor ? 0.75 : 0,
                  }}
                  aria-hidden="true"
                />

                {/* Pos */}
                <div className="text-center text-sm font-semibold tabular-nums text-neutral-900 dark:text-neutral-100">
                  {pos}
                </div>

                {/* Driver (clickable) + TEAM UNDER NAME ALWAYS */}
                <button
                  type="button"
                  onClick={() => onDriverClick?.(row)}
                  className={[
                    "min-w-0 text-left rounded-lg",
                    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[rgb(var(--accent))]",
                  ].join(" ")}
                  title="Open driver details"
                >
                  <div className="min-w-0 flex items-center gap-2">
                    <div className="truncate text-sm font-semibold text-neutral-900 dark:text-neutral-100">
                      {driver}
                    </div>
                    <span
                      className={[
                        "shrink-0",
                        "text-[18px] leading-none",
                        "text-neutral-600 dark:text-neutral-300",
                        "opacity-90",
                      ].join(" ")}
                      aria-hidden="true"
                      title="More info"
                    >
                      ⓘ
                    </span>
                  </div>

                  <div className="truncate text-xs text-neutral-500 dark:text-neutral-400">
                    {team}
                  </div>
                </button>

                {/* Right-side columns */}
                {viewType === "race" ? (
                  <>
                    <div className="text-right text-sm font-mono text-neutral-700 dark:text-neutral-300">
                      {time}
                    </div>
                    <div className="text-right text-sm font-semibold tabular-nums">
                      <span
                        className={
                          pts > 0
                            ? "text-emerald-600 dark:text-emerald-400"
                            : "text-neutral-400 dark:text-neutral-600"
                        }
                      >
                        {ptsText}
                      </span>
                    </div>
                  </>
                ) : null}

                {viewType === "qualifying" ? (
                  <>
                    <div className="text-center text-sm font-mono text-neutral-600 dark:text-neutral-400">
                      {q1}
                    </div>
                    <div className="text-center text-sm font-mono text-neutral-600 dark:text-neutral-400">
                      {q2}
                    </div>
                    <div className="text-center text-sm font-mono font-semibold text-neutral-900 dark:text-neutral-100">
                      {q3}
                    </div>
                  </>
                ) : null}
              </div>
            );
          })
        )}
      </div>
    </Card>
  );
}