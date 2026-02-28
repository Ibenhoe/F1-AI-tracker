// src/components/wiki/components/CircuitCard.jsx
import Card from "../../ui/Card";

export default function CircuitCard({
  tab = "overview", // "overview" | "logo"
  circuitInfo,
  circuitImage,
}) {
  if (!circuitInfo) return null;

  const name = circuitInfo.name ?? "Circuit";
  const location = circuitInfo.location ?? "—";
  const overview = circuitInfo.history ?? "—";
  const lapRecord = circuitInfo.lapRecord ?? "—";

  // --- LOGO TAB ---
  if (tab === "logo") {
    return (
      <div className="h-full min-h-0 overflow-hidden">
        <Card className="h-full p-4 flex items-center justify-center" clip bordered>
          {circuitImage ? (
            <img
              src={circuitImage}
              alt={`${name} logo`}
              className="max-h-[240px] max-w-full object-contain"
              draggable={false}
            />
          ) : (
            <div className="text-sm text-neutral-500 dark:text-neutral-400">
              No image available.
            </div>
          )}
        </Card>
      </div>
    );
  }

  // --- OVERVIEW TAB ---
  return (
    <div className="h-full min-h-0 overflow-hidden">
      <Card className="h-full p-4" clip bordered>
        <div className="flex h-full min-h-0 flex-col">
          {/* Top content */}
          <div className="min-w-0">
            <h2 className="text-base font-semibold tracking-tight text-neutral-900 dark:text-neutral-50 line-clamp-1">
              {name}
            </h2>

            <p className="mt-1 text-sm text-neutral-500 dark:text-neutral-400 line-clamp-1">
              {location}
            </p>

            {/* show a bit more text to reduce empty space */}
            <p className="mt-3 text-sm leading-relaxed text-neutral-700 dark:text-neutral-300 line-clamp-6">
              {overview}
            </p>
          </div>

          {/* Bottom (sticks to bottom) */}
          <div className="mt-auto pt-3">
            <div className="h-px w-full bg-black/5 dark:bg-white/10" />

            <div className="mt-3 flex items-center justify-between gap-4">
              <div className="flex items-center gap-2">
                <span
                  className={[
                    "inline-block h-2 w-2 rounded-full",
                    "bg-[rgb(var(--accent))]",
                    "opacity-80",
                  ].join(" ")}
                  aria-hidden="true"
                />
                <span className="text-[11px] font-semibold uppercase tracking-widest text-neutral-500 dark:text-neutral-500">
                  Lap record
                </span>
              </div>

              <span className="text-sm font-semibold tabular-nums text-neutral-900 dark:text-neutral-50">
                {lapRecord}
              </span>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}