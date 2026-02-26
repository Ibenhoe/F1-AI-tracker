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

function StatRow({ icon: Icon, label, value }) {
  return (
    <div className="flex items-center justify-between py-2">
      <div className="flex items-center gap-2 text-xs text-neutral-500 dark:text-neutral-400">
        <Icon size={14} className="opacity-80" />
        <span className="font-medium">{label}</span>
      </div>
      <div className="text-sm font-medium tabular-nums text-neutral-900 dark:text-neutral-50">
        {value}
      </div>
    </div>
  );
}

export default function WeatherWidget({ data }) {
  if (!data) {
    return <div className="h-full" />;
  }

  const condition = normalizeCondition(data.condition);
  const Icon = getIcon(condition);

  const air = clampNum(data.temp);
  const track = clampNum(data.trackTemp);
  const humidity = clampNum(data.humidity);
  const wind = clampNum(data.windSpeed);

  // Keep as you wanted: no condition text
  const SHOW_CONDITION = false;

  return (
    <div className="h-full min-h-0">
      {/* Single calm surface */}
      <div className="flex h-full min-h-0 flex-col rounded-2xl bg-white/70 ring-1 ring-neutral-200/70 px-5 py-5 backdrop-blur-sm dark:bg-[rgb(var(--panel))] dark:ring-white/10 dark:backdrop-blur-none">
        {/* Top summary */}
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="flex items-end gap-2">
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
                  <span className="text-neutral-300 dark:text-white/20"> · </span>
                  <span className="text-neutral-600 dark:text-neutral-300">
                    {condition}
                  </span>
                </>
              ) : null}
            </div>
          </div>

          {/* Icon, no glow/chip */}
          <Icon
            size={26}
            className="mt-1 shrink-0 text-neutral-700 dark:text-neutral-200"
          />
        </div>

        {/* Divider */}
        <div className="my-4 h-px w-full bg-neutral-200/70 dark:bg-white/10" />

        {/* Secondary stats as list rows */}
        <div className="divide-y divide-neutral-200/70 dark:divide-white/10">
          <StatRow
            icon={Droplets}
            label="Humidity"
            value={`${Math.round(humidity)}%`}
          />
          <StatRow
            icon={Wind}
            label="Wind"
            value={`${wind.toFixed(1)} km/h`}
          />
          <StatRow
            icon={Thermometer}
            label="Track"
            value={`${Math.round(track)}°C`}
          />
        </div>
      </div>
    </div>
  );
}