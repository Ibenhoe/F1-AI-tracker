import { useMemo, useState } from "react";
import { Pause, Play, Gauge } from "lucide-react";

import apiClient from "../services/apiClient";
import Button from "./ui/Button.jsx";
import Badge from "./ui/Badge.jsx";

const SPEEDS = [
  { value: 0.5, label: "0.5×", hint: "Slow" },
  { value: 1.0, label: "1.0×", hint: "Normal" },
  { value: 2.0, label: "2.0×", hint: "Fast" },
  { value: 4.0, label: "4.0×", hint: "Very fast" },
];

function statusFor({ connected, raceInitialized, raceRunning }) {
  if (!connected) return { label: "Connecting", variant: "warning" };
  if (connected && !raceInitialized) return { label: "Loading race", variant: "warning" };
  if (connected && raceInitialized && !raceRunning) return { label: "Ready", variant: "success" };
  return { label: "Running", variant: "warning" };
}

export default function RaceControls({ raceInitialized, raceRunning, connected }) {
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

  return (
    <div className="space-y-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold tracking-tight">Race controls</h3>
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
          className="justify-center border border-neutral-800"
        >
          <Play size={16} />
          Resume
        </Button>
      </div>

      {/* Speed */}
      <div className="rounded-xl border border-neutral-800 bg-neutral-950/40 p-3">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Gauge size={16} className="text-neutral-500" />
            <div>
              <div className="text-xs font-medium text-neutral-300">
                Simulation speed
              </div>
            </div>
          </div>

          <select
            value={speed}
            onChange={(e) => handleSpeedChange(parseFloat(e.target.value))}
            disabled={!connected}
            className={[
              "w-40 rounded-lg border px-3 py-2 text-sm",
              "border-neutral-800 bg-neutral-950/40 text-neutral-100",
              "focus:outline-none focus:ring-2 focus:ring-neutral-700",
              !connected ? "opacity-60" : "hover:border-neutral-700",
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

      {/* Connection hint */}
      {!connected ? (
        <div className="text-xs text-neutral-500">
          Waiting for backend connection…
        </div>
      ) : null}
    </div>
  );
}
