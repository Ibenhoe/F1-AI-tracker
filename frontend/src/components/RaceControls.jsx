import { useMemo, useState } from "react";
import { Pause, Play, Gauge, Flag } from "lucide-react";

import apiClient from "../services/apiClient";
import Button from "./ui/Button.jsx";
import Badge from "./ui/Badge.jsx";

const SPEEDS = [
  { value: 1.0, label: "1×", hint: "Normal" },
  { value: 2.0, label: "2×", hint: "Fast" },
  { value: 3.0, label: "3×", hint: "Very fast" },
  { value: 4.0, label: "4×", hint: "Turbo" },
];

function statusFor({ connected, raceInitialized, raceRunning }) {
  if (!connected) return { label: "Connecting", variant: "warning" };
  if (connected && !raceInitialized) return { label: "Loading race", variant: "warning" };
  if (connected && raceInitialized && !raceRunning) return { label: "Ready", variant: "success" };
  return { label: "Running", variant: "warning" };
}

function clamp01(n) {
  if (!Number.isFinite(n)) return 0;
  return Math.min(1, Math.max(0, n));
}

export default function RaceControls({
  raceInitialized,
  raceRunning,
  connected,
  raceData, // <-- NEW (optional): { race, currentLap, totalLaps }
}) {
  const [speed, setSpeed] = useState(1.0);

  const status = useMemo(
    () => statusFor({ connected, raceInitialized, raceRunning }),
    [connected, raceInitialized, raceRunning]
  );

  const canStart = connected && raceInitialized && !raceRunning;
  const canPause = connected && raceRunning;
  const canResume = connected && raceInitialized && !raceRunning;

  const handleStart = () => {
    if (canStart) apiClient.startRace(speed);
  };

  const handlePause = () => {
    if (canPause) apiClient.pauseRace();
  };

  const handleResume = () => {
    if (canResume) apiClient.resumeRace();
  };

  const handleSpeedChange = (newSpeed) => {
    setSpeed(newSpeed);
    if (connected) apiClient.setSimulationSpeed(newSpeed);
  };

  const total = Number(raceData?.totalLaps ?? 0);
  const current = Number(raceData?.currentLap ?? 0);
  const progress = clamp01(total > 0 ? current / total : 0);
  const progressPct = Math.round(progress * 100);

  return (
    <div className="space-y-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold tracking-tight">Race controls</h3>
          {raceData?.race ? (
            <div className="mt-1 text-xs text-neutral-600 dark:text-neutral-400">
              {raceData.race}
            </div>
          ) : null}
        </div>
        <Badge variant={status.variant}>{status.label}</Badge>
      </div>

      {/* Buttons */}
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
        <Button
          variant="primary"
          onClick={handleStart}
          disabled={!canStart}
          className="justify-center"
        >
          <Play size={16} />
          Start
        </Button>

        <Button
          variant="secondary"
          onClick={handlePause}
          disabled={!canPause}
          className="justify-center"
        >
          <Pause size={16} />
          Pause
        </Button>

        <Button
          variant="ghost"
          onClick={handleResume}
          disabled={!canResume}
          className="justify-center border border-neutral-300 dark:border-neutral-800"
        >
          <Play size={16} />
          Resume
        </Button>
      </div>

      {/* Speed */}
      <div className="rounded-xl border border-neutral-200 bg-neutral-50 p-3 dark:border-neutral-800 dark:bg-neutral-950/40">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Gauge size={16} className="text-neutral-500" />
            <div className="text-xs font-medium text-neutral-700 dark:text-neutral-300">
              Simulation speed
            </div>
          </div>

          <select
            value={speed}
            onChange={(e) => handleSpeedChange(parseFloat(e.target.value))}
            disabled={!connected}
            className={[
              "w-40 rounded-lg border px-3 py-2 text-sm",
              "border-neutral-300 bg-white text-neutral-900",
              "focus:outline-none focus:ring-2 focus:ring-neutral-300",
              "dark:border-neutral-800 dark:bg-neutral-950/40 dark:text-neutral-100 dark:focus:ring-neutral-700",
              !connected ? "opacity-60" : "hover:border-neutral-400 dark:hover:border-neutral-700",
            ].join(" ")}
          >
            {SPEEDS.map((s) => (
              <option key={s.value} value={s.value}>
                {s.label} ({s.hint})
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Lap progress (moved here) */}
      <div className="rounded-xl border border-neutral-200 bg-neutral-50 p-3 dark:border-neutral-800 dark:bg-neutral-950/40">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Flag size={16} className="text-neutral-500" />
            <span className="text-xs font-medium text-neutral-700 dark:text-neutral-300">
              Lap progress
            </span>
          </div>

          <div className="flex items-center gap-2">
            <Badge variant="neutral">
              <span className="tabular-nums">
                {current}/{total || "—"}
              </span>
            </Badge>
            <span className="text-xs text-neutral-700 dark:text-neutral-300 tabular-nums">
              {total > 0 ? `${progressPct}%` : "—"}
            </span>
          </div>
        </div>

        <div className="mt-3 h-2 w-full rounded-full bg-neutral-200 dark:bg-neutral-900">
          <div
            className="h-2 rounded-full bg-neutral-900 dark:bg-neutral-100"
            style={{ width: `${total > 0 ? progressPct : 0}%` }}
          />
        </div>

        <div className="mt-2 flex items-center justify-between text-[11px] text-neutral-600 dark:text-neutral-500">
          <span>Start</span>
          <span>Finish</span>
        </div>
      </div>

      {!connected ? (
        <div className="text-xs text-neutral-600 dark:text-neutral-500">
          Waiting for backend connection…
        </div>
      ) : null}
    </div>
  );
}
