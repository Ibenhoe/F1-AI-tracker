import { Wind, Droplets, Thermometer, Cloud, CloudRain, Sun } from "lucide-react";

function clampNum(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function normalizeCondition(raw) {
  if (raw === true) return "Dry";
  if (raw === false) return "Wet";

  if (typeof raw === "string") {
    const s = raw.trim();
    if (!s) return "Unknown";
    if (s.toLowerCase() === "false") return "Wet";
    if (s.toLowerCase() === "true") return "Dry";
    return s;
  }

  return "Unknown";
}

function getIcon(condition) {
  const c = condition.toLowerCase();
  if (c.includes("rain") || c.includes("wet")) return CloudRain;
  if (c.includes("dry") || c.includes("clear") || c.includes("sun")) return Sun;
  return Cloud;
}

function StatTile({ icon: Icon, label, value }) {
  return (
    <div
      className={[
        "rounded-2xl px-4 py-3",
        // light
        "bg-white ring-1 ring-neutral-200/70",
        "shadow-[0_1px_0_rgba(0,0,0,0.03),0_10px_28px_rgba(0,0,0,0.08)]",
        // dark
        "dark:bg-neutral-950/30 dark:ring-white/10",
        "dark:shadow-[0_1px_0_rgba(255,255,255,0.04),0_18px_50px_rgba(0,0,0,0.45)]",
      ].join(" ")}
    >
      <div className="flex items-center gap-2 text-[10px] uppercase tracking-widest text-neutral-500 dark:text-neutral-400">
        <Icon size={13} />
        {label}
      </div>

      <div className="mt-1 text-base font-semibold tabular-nums text-neutral-900 dark:text-neutral-50">
        {value}
      </div>
    </div>
  );
}

export default function WeatherWidget({ data }) {
  if (!data) {
    return (
      <div
        className={[
          "h-full rounded-2xl",
          "bg-white ring-1 ring-neutral-200/70",
          "dark:bg-white/5 dark:ring-white/10",
        ].join(" ")}
      />
    );
  }

  const condition = normalizeCondition(data.condition);
  const Icon = getIcon(condition);

  const air = clampNum(data.temp);
  const track = clampNum(data.trackTemp);
  const humidity = clampNum(data.humidity);
  const wind = clampNum(data.windSpeed);

  // You said you don't want condition shown
  const SHOW_CONDITION = false;

  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      <div
        className={[
          "flex-1 min-h-0",
          "rounded-2xl px-6 py-6",
          // light
          "bg-white ring-1 ring-neutral-200/70",
          "shadow-[0_1px_0_rgba(0,0,0,0.03),0_14px_40px_rgba(0,0,0,0.10)]",
          // dark
          "dark:bg-neutral-950/30 dark:ring-white/10",
          "dark:shadow-[0_1px_0_rgba(255,255,255,0.04),0_18px_50px_rgba(0,0,0,0.55)]",
        ].join(" ")}
      >
        <div className="flex h-full flex-col justify-between">
          <div className="flex items-start justify-between gap-6">
            <div className="min-w-0">
              <div className="flex items-end gap-3">
                <div className="text-5xl font-semibold tracking-tight tabular-nums leading-none text-neutral-900 dark:text-neutral-50">
                  {Math.round(air)}°
                </div>
                <div className="pb-1 text-sm text-neutral-500 dark:text-neutral-400">
                  air
                </div>
              </div>

              <div className="mt-3 text-sm text-neutral-600 dark:text-neutral-400">
                Track{" "}
                <span className="font-semibold tabular-nums text-neutral-900 dark:text-neutral-50">
                  {Math.round(track)}°C
                </span>

                {SHOW_CONDITION ? (
                  <>
                    <span className="text-neutral-400 dark:text-neutral-600"> · </span>
                    <span className="text-neutral-700 dark:text-neutral-300">
                      {condition}
                    </span>
                  </>
                ) : null}
              </div>
            </div>

            <div
              className={[
                "grid h-14 w-14 place-items-center rounded-2xl",
                //light
                "bg-neutral-50 ring-1 ring-neutral-200/70",
                //dark
                "dark:bg-white/5 dark:ring-white/10",
              ].join(" ")}
            >
              <Icon className="opacity-90 text-neutral-900 dark:text-neutral-50" size={24} />
            </div>
          </div>

          {/* divider for spacing / fill */}
          <div className="mt-6 h-px w-full bg-neutral-200/70 dark:bg-white/10" />
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2">
        <StatTile icon={Droplets} label="Humidity" value={`${Math.round(humidity)}%`} />
        <StatTile icon={Wind} label="Wind" value={`${wind.toFixed(1)} km/h`} />
        <StatTile icon={Thermometer} label="Track" value={`${Math.round(track)}°C`} />
      </div>
    </div>
  );
}