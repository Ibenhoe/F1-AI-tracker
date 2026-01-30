import { Flag, Gauge } from "lucide-react";
import Badge from "./ui/Badge.jsx";

function clamp01(n) {
  if (!Number.isFinite(n)) return 0;
  return Math.min(1, Math.max(0, n));
}

export default function RaceInfo({ data }) {
  if (!data) {
    return (
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold tracking-tight">Race</h2>
          <Badge variant="neutral">Loading</Badge>
        </div>
        <div className="h-24 rounded-xl border border-neutral-800 bg-neutral-950/40" />
        <div className="h-10 rounded-lg border border-neutral-800 bg-neutral-950/40" />
      </div>
    );
  }

  const total = Number(data.totalLaps ?? 0);
  const current = Number(data.currentLap ?? 0);
  const progress = clamp01(total > 0 ? current / total : 0);
  const progressPct = Math.round(progress * 100);

  return (
    <div className="space-y-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold tracking-tight">Race</h2>
          <p className="mt-1 text-xs text-neutral-400">{data.race}</p>
        </div>
        <Badge variant="neutral">
          {current}/{total} laps
        </Badge>
      </div>

      <div className="rounded-xl border border-neutral-800 bg-neutral-950/40 p-4">
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Flag size={16} className="text-neutral-500" />
            <span className="text-xs text-neutral-400">Lap progress</span>
          </div>
          <div className="flex items-center gap-2">
            <Gauge size={16} className="text-neutral-500" />
            <span className="text-xs text-neutral-300">{progressPct}%</span>
          </div>
        </div>

        <div className="mt-3 h-2 w-full rounded-full bg-neutral-900">
          <div
            className="h-2 rounded-full bg-neutral-100"
            style={{ width: `${progressPct}%` }}
          />
        </div>

        <div className="mt-3 flex items-center justify-between text-xs text-neutral-500">
          <span>Start</span>
          <span>Finish</span>
        </div>
      </div>
    </div>
  );
}
