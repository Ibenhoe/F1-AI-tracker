import { Wind, Droplets, Thermometer } from "lucide-react";
import Badge from "./ui/Badge.jsx";

function Stat({ icon: Icon, label, value }) {
  return (
    <div className="flex items-center justify-between rounded-lg border border-neutral-800 bg-neutral-950/40 px-3 py-2">
      <div className="flex items-center gap-2">
        <Icon size={14} className="text-neutral-500" />
        <span className="text-xs text-neutral-400">{label}</span>
      </div>
      <span className="text-sm font-medium text-neutral-100">{value}</span>
    </div>
  );
}

export default function WeatherWidget({ data }) {
  if (!data) {
    return (
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold tracking-tight">Weather</h2>
          <Badge variant="neutral">Loading</Badge>
        </div>
        <div className="h-24 rounded-lg border border-neutral-800 bg-neutral-950/40" />
        <div className="grid grid-cols-1 gap-2">
          <div className="h-10 rounded-lg border border-neutral-800 bg-neutral-950/40" />
          <div className="h-10 rounded-lg border border-neutral-800 bg-neutral-950/40" />
          <div className="h-10 rounded-lg border border-neutral-800 bg-neutral-950/40" />
        </div>
      </div>
    );
  }

    const raw = data.condition;

  let condition = "Unknown";
  if (typeof raw === "string" && raw.trim().length > 0) condition = raw.trim();
  else if (raw === true) condition = "Dry";
  else if (raw === false) condition = "Wet";

  const c = condition.toLowerCase();
  const badgeVariant =
    c.includes("dry") || c.includes("clear") ? "success" : c.includes("wet") || c.includes("rain") ? "warning" : "neutral";


  return (
    <div className="space-y-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold tracking-tight">Weather</h2>
          <p className="mt-1 text-xs text-neutral-400">
            Track-side conditions snapshot
          </p>
        </div>
        <Badge variant={badgeVariant}>{condition}</Badge>
      </div>

      <div className="flex items-end justify-between gap-4 rounded-xl border border-neutral-800 bg-neutral-950/40 p-4">
        <div>
          <div className="text-4xl font-semibold leading-none tracking-tight">
            {data.temp}°
          </div>
          <div className="mt-2 text-xs text-neutral-400">Air temperature</div>
        </div>

        <div className="text-right">
          <div className="text-sm font-medium text-neutral-100">
            {data.trackTemp}°C
          </div>
          <div className="mt-1 text-xs text-neutral-400">Track temperature</div>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-2">
        <Stat icon={Droplets} label="Humidity" value={`${data.humidity}%`} />
        <Stat icon={Wind} label="Wind" value={`${data.windSpeed} km/h`} />
        <Stat
          icon={Thermometer}
          label="Track temp"
          value={`${data.trackTemp}°C`}
        />
      </div>
    </div>
  );
}
