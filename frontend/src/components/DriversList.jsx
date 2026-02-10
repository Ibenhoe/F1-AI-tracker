import { ArrowDown, ArrowUp, Minus } from "lucide-react";
import Badge from "./ui/Badge.jsx";

function tireMeta(compound) {
  const c = String(compound ?? "").toUpperCase();

  if (c === "SOFT")
    return {
      label: "S",
      ring: "ring-red-500/30 dark:ring-red-500/30",
      bg: "bg-red-500/10 dark:bg-red-500/10",
      text: "text-red-700 dark:text-red-200",
    };

  if (c === "MEDIUM")
    return {
      label: "M",
      ring: "ring-amber-500/30 dark:ring-amber-500/30",
      bg: "bg-amber-500/10 dark:bg-amber-500/10",
      text: "text-amber-800 dark:text-amber-200",
    };

  if (c === "HARD")
    return {
      label: "H",
      ring: "ring-neutral-400/30 dark:ring-neutral-400/30",
      bg: "bg-neutral-200/60 dark:bg-neutral-200/10",
      text: "text-neutral-800 dark:text-neutral-200",
    };

  if (c === "INTERMEDIATE")
    return {
      label: "I",
      ring: "ring-emerald-500/30 dark:ring-emerald-500/30",
      bg: "bg-emerald-500/10 dark:bg-emerald-500/10",
      text: "text-emerald-800 dark:text-emerald-200",
    };

  if (c === "WET")
    return {
      label: "W",
      ring: "ring-blue-500/30 dark:ring-blue-500/30",
      bg: "bg-blue-500/10 dark:bg-blue-500/10",
      text: "text-blue-800 dark:text-blue-200",
    };

  return {
    label: "?",
    ring: "ring-neutral-300 dark:ring-neutral-700",
    bg: "bg-neutral-100 dark:bg-neutral-900",
    text: "text-neutral-700 dark:text-neutral-300",
  };
}

function PosDelta({ value }) {
  const v = Number(value ?? 0);

  if (v > 0) {
    return (
      <span className="inline-flex items-center gap-1 text-xs font-medium text-emerald-700 dark:text-emerald-300">
        <ArrowUp size={14} />
        +{v}
      </span>
    );
  }
  if (v < 0) {
    return (
      <span className="inline-flex items-center gap-1 text-xs font-medium text-red-700 dark:text-red-300">
        <ArrowDown size={14} />
        {v}
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 text-xs font-medium text-neutral-500 dark:text-neutral-400">
      <Minus size={14} />
      0
    </span>
  );
}

export default function DriversList({ drivers, currentLap }) {
  return (
    <div className="space-y-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold tracking-tight">Standings</h2>
          <p className="mt-1 text-xs text-neutral-600 dark:text-neutral-400">
            Current classification and tire status
          </p>
        </div>
        <Badge variant="neutral">Lap {currentLap || 0}</Badge>
      </div>

      <div className="divide-y divide-neutral-200 overflow-hidden rounded-xl border border-neutral-200 dark:divide-neutral-900 dark:border-neutral-800">
        {drivers && drivers.length > 0 ? (
          drivers.map((driver) => {
            const meta = tireMeta(driver.tire_compound);
            const code = driver.driver_code || String(driver.position ?? "");
            const name = driver.driver_name || driver.driver_code || "Unknown";
            const team = driver.team || "—";

            return (
              <div
                key={code}
                className={[
                  "flex items-center gap-3 px-3 py-3",
                  "bg-white hover:bg-neutral-50",
                  "dark:bg-neutral-950/40 dark:hover:bg-neutral-950/60",
                ].join(" ")}
              >
                {/* Position */}
                <div className="w-8 shrink-0 text-center text-sm font-semibold text-neutral-900 dark:text-neutral-100">
                  {driver.position ?? "—"}
                </div>

                {/* Driver */}
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm font-medium text-neutral-900 dark:text-neutral-100">
                    {name}
                  </div>
                  <div className="truncate text-xs text-neutral-600 dark:text-neutral-500">
                    {team}
                  </div>
                </div>

                {/* Lap time */}
                <div className="hidden w-24 shrink-0 text-right sm:block">
                  <div className="text-xs text-neutral-600 dark:text-neutral-500">
                    Lap
                  </div>
                  <div className="text-sm font-medium text-neutral-900 dark:text-neutral-100 tabular-nums">
                    {driver.lap_time || "--:--"}
                  </div>
                </div>

                {/* Tires */}
                <div className="w-20 shrink-0 text-right">
                  <div className="text-xs text-neutral-600 dark:text-neutral-500">
                    Tires
                  </div>
                  <div className="mt-1 inline-flex items-center justify-end gap-2">
                    <span
                      className={[
                        "inline-flex h-6 w-6 items-center justify-center rounded-full text-xs font-semibold",
                        meta.bg,
                        meta.text,
                        "ring-1",
                        meta.ring,
                      ].join(" ")}
                      title={driver.tire_compound || "Unknown"}
                    >
                      {meta.label}
                    </span>
                    <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100 tabular-nums">
                      {driver.tire_age ?? 0}
                    </span>
                  </div>
                </div>

                {/* Pit + delta */}
                <div className="w-20 shrink-0 text-right">
                  <div className="text-xs text-neutral-600 dark:text-neutral-500">
                    Pit
                  </div>
                  <div className="mt-1 flex items-center justify-end gap-2">
                    <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100 tabular-nums">
                      {driver.pit_stops ?? 0}
                    </span>
                    <PosDelta value={driver.position_change} />
                  </div>
                </div>
              </div>
            );
          })
        ) : (
          <div className="bg-neutral-50 px-4 py-10 text-center dark:bg-neutral-950/40">
            <div className="text-sm font-medium text-neutral-900 dark:text-neutral-200">
              Waiting for race data…
            </div>
            <div className="mt-1 text-xs text-neutral-600 dark:text-neutral-500">
              Standings will appear once telemetry is available.
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
