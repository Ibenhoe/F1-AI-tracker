import React from "react";
import { getTeamColor } from "../../../utils/teamColors";
import { normalizeDriver } from "../utils/telemetry";

const DRIVER_TEAMS_FALLBACK = {
  VER: "Red Bull Racing",
  PER: "Red Bull Racing",
  HAM: "Mercedes",
  RUS: "Mercedes",
  LEC: "Ferrari",
  SAI: "Ferrari",
  NOR: "McLaren",
  PIA: "McLaren",
  ALO: "Aston Martin",
  STR: "Aston Martin",
  GAS: "Alpine",
  OCO: "Alpine",
  TSU: "RB",
  RIC: "RB",
  LAW: "RB",
  ALB: "Williams",
  SAR: "Williams",
  COL: "Williams",
  HUL: "Haas F1 Team",
  MAG: "Haas F1 Team",
  BEA: "Haas F1 Team",
  BOT: "Kick Sauber",
  ZHO: "Kick Sauber",
};

function getDriverTeamName(code, driver) {
  const hasConstructorKey = Object.prototype.hasOwnProperty.call(driver || {}, "constructor");
  const raw =
    driver?.team ||
    driver?.team_name ||
    (hasConstructorKey ? driver?.constructor : null) ||
    driver?.constructor_name;

  if (raw && raw !== "Unknown") return raw;
  return DRIVER_TEAMS_FALLBACK[code] || "";
}

function tireMeta(compound) {
  const c = String(compound ?? "").toUpperCase();
  if (c === "SOFT")
    return { label: "S", color: "#ef4444", bg: "bg-red-500/10", text: "text-red-600 dark:text-red-400" };
  if (c === "MEDIUM")
    return { label: "M", color: "#f59e0b", bg: "bg-amber-500/10", text: "text-amber-600 dark:text-amber-400" };
  if (c === "HARD")
    return { label: "H", color: "#a3a3a3", bg: "bg-neutral-300/30", text: "text-neutral-700 dark:text-neutral-300" };
  if (c === "INTERMEDIATE")
    return { label: "I", color: "#10b981", bg: "bg-emerald-500/10", text: "text-emerald-600 dark:text-emerald-400" };
  if (c === "WET")
    return { label: "W", color: "#3b82f6", bg: "bg-blue-500/10", text: "text-blue-600 dark:text-blue-400" };
  return { label: "?", color: "#888", bg: "bg-neutral-200/40", text: "text-neutral-500" };
}

function TelBar({ label, value, color = "#ef4444" }) {
  const pct = Math.min(100, Math.max(0, Number(value) || 0));
  return (
    <div className="flex items-center gap-2">
      <span className="w-16 flex-shrink-0 text-[11px] font-semibold uppercase tracking-wider text-neutral-500 dark:text-neutral-400">
        {label}
      </span>
      <div className="flex-1 h-1.5 rounded-full bg-neutral-200 dark:bg-neutral-800 overflow-hidden">
        <div className="h-full rounded-full transition-all duration-100" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="w-10 text-right text-xs tabular-nums font-semibold text-neutral-800 dark:text-neutral-200">
        {Math.round(pct)}%
      </span>
    </div>
  );
}

const DriverInfoPanel = ({ driver, driverData, showTelemetry }) => {
  if (!driver || !driverData) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-neutral-500 dark:text-neutral-400">
        Select a driver to view details
      </div>
    );
  }

  // ✅ Single source of truth for formatting/units (matches TrackRenderer)
  const t = normalizeDriver(driverData);

  const team = getDriverTeamName(driver, driverData);
  const teamColor = getTeamColor(team) || "rgb(var(--accent))";
  const tire = tireMeta(t.tireCompound);
  const isDNF = t.status === "OUT";
  const drsOn = t.drsOn;

  return (
    <div className="flex flex-col gap-4">
      <div
        className="relative overflow-hidden rounded-xl border border-neutral-200/80 bg-white/90 p-4 dark:border-white/10 dark:bg-neutral-950/60"
        style={{ borderColor: `${teamColor}40` }}
      >
        <div className="absolute inset-x-0 top-0 h-1 rounded-t-xl" style={{ backgroundColor: teamColor }} />
        <div className="mt-1 flex items-start justify-between gap-3">
          <div>
            <div className="text-lg font-semibold tracking-tight text-neutral-900 dark:text-neutral-50">
              {t.driverName || driver}
            </div>
            <div className="mt-0.5 text-[11px] font-semibold uppercase tracking-widest text-neutral-500 dark:text-neutral-400">
              {team || "—"}
            </div>
          </div>
          <div className="text-right">
            <div className="text-2xl font-bold tabular-nums text-neutral-900 dark:text-neutral-50">
              P{t.position}
            </div>
            {isDNF && <div className="mt-0.5 text-xs font-semibold text-red-500">DNF</div>}
          </div>
        </div>
      </div>

      {showTelemetry && (
        <div className="rounded-xl border border-neutral-200/80 bg-white/70 p-4 dark:border-white/10 dark:bg-neutral-950/50">
          <div className="mb-3 text-[11px] font-semibold uppercase tracking-widest text-neutral-500 dark:text-neutral-400">
            Telemetry
          </div>
          <div className="flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-neutral-500 dark:text-neutral-400">
                Speed
              </span>
              <span className="text-sm font-bold tabular-nums text-neutral-900 dark:text-neutral-100">
                {Number(t.speedKmh ?? 0).toFixed(0)}{" "}
                <span className="font-normal text-xs text-neutral-500">km/h</span>
              </span>
            </div>

            <TelBar label="Throttle" value={t.throttlePct} color="#22c55e" />
            <TelBar label="Brake" value={t.brakePct} color="#ef4444" />

            <div className="flex items-center justify-between">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-neutral-500 dark:text-neutral-400">
                DRS
              </span>
              <span
                className={`rounded-full px-2 py-0.5 text-xs font-bold ${drsOn
                    ? "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400"
                    : "bg-neutral-200/60 text-neutral-500 dark:bg-neutral-800 dark:text-neutral-400"
                  }`}
              >
                {drsOn ? "ON" : "OFF"}
              </span>
            </div>
          </div>
        </div>
      )}

      <div className="rounded-xl border border-neutral-200/80 bg-white/70 p-4 dark:border-white/10 dark:bg-neutral-950/50">
        <div className="mb-3 text-[11px] font-semibold uppercase tracking-widest text-neutral-500 dark:text-neutral-400">
          Tyre
        </div>
        <div className="flex items-center gap-3">
          <div
            className={`flex h-10 w-10 items-center justify-center rounded-full text-base font-black ring-2 ${tire.bg} ${tire.text}`}
            style={{ ringColor: teamColor }}
            title={t.tireCompound || "?"}
          >
            {tire.label}
          </div>
          <div>
            <div className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">
              {t.tireCompound || "—"}
            </div>
            <div className="mt-0.5 text-xs text-neutral-500 dark:text-neutral-400">
              {t.tireAge ?? 0} laps on set
            </div>
          </div>
        </div>
      </div>

      <div className="rounded-xl border border-neutral-200/80 bg-white/70 p-4 dark:border-white/10 dark:bg-neutral-950/50">
        <div className="mb-3 text-[11px] font-semibold uppercase tracking-widest text-neutral-500 dark:text-neutral-400">
          Race info
        </div>
        <div className="flex flex-col gap-2 text-sm">
          <div className="flex items-center justify-between">
            <span className="text-neutral-500 dark:text-neutral-400">Gap</span>
            <span className="tabular-nums font-medium text-neutral-900 dark:text-neutral-100">
              {t.position === 1 ? (
                <span className="font-bold text-yellow-500">LEAD</span>
              ) : (
                t.gap ?? "—"
              )}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-neutral-500 dark:text-neutral-400">Pit stops</span>
            <span className="tabular-nums font-medium text-neutral-900 dark:text-neutral-100">
              {t.pitStops ?? 0}
            </span>
          </div>
          {t.lapTime && (
            <div className="flex items-center justify-between">
              <span className="text-neutral-500 dark:text-neutral-400">Last lap</span>
              <span className="tabular-nums font-medium text-neutral-900 dark:text-neutral-100">
                {t.lapTime}
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DriverInfoPanel;