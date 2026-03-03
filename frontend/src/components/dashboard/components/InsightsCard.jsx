// src/components/dashboard/components/InsightsCard.jsx
import { useEffect, useMemo, useState, useCallback } from "react";

import SegmentedControl from "./SegmentedControl";
import EmptyState from "./EmptyState";

import PredictionsPanel from "../panels/PredictionsPanel";
import BattlesWidget from "../panels/BattlesWidget";
import WeatherWidget from "../panels/WeatherWidget";
import NotificationsPanel from "../panels/NotificationsPanel";

const TAB_SET = new Set(["predictions", "battles", "weather", "notifications"]);

export default function InsightsCard({
  predictions,
  currentLap,
  modelMetrics,
  totalLaps,
  weatherData,
  notifications,
  drivers,
  selectedRace,
}) {
  const [tab, _setTab] = useState("predictions");
  const notifCount = Array.isArray(notifications) ? notifications.length : 0;

  // Only allow valid tabs (prevents invalid state without effect "repair")
  const setTab = useCallback((next) => {
    const v = String(next || "");
    _setTab(TAB_SET.has(v) ? v : "predictions");
  }, []);

  // Reset tab when switching race (prevents weird empty panels after race switch)
  useEffect(() => {
    if (typeof selectedRace === "number") _setTab("predictions");
  }, [selectedRace]);

  const items = useMemo(
    () => [
      { id: "predictions", label: "Predictions" },
      { id: "battles", label: "Battles" },
      { id: "weather", label: "Weather" },
      {
        id: "notifications",
        label: "Notifications",
        trailing:
          notifCount > 0 ? (
            <span
              className={[
                "inline-flex items-center justify-center",
                "h-5 min-w-5 px-1.5 rounded-full",
                "text-[11px] font-bold tabular-nums leading-none",
                "bg-[rgb(var(--accent))] text-[rgb(var(--accent-fg))]",
                "ring-1 ring-black/5 dark:ring-white/10 opacity-95",
              ].join(" ")}
              aria-label={`${notifCount} notifications`}
              title={`${notifCount} notifications`}
            >
              {notifCount > 99 ? "99+" : String(notifCount)}
            </span>
          ) : null,
      },
    ],
    [notifCount]
  );

  const hasPredictions = Array.isArray(predictions) && predictions.length > 0;
  const hasDrivers = Array.isArray(drivers) && drivers.length > 0;
  const hasNotifications = Array.isArray(notifications) && notifications.length > 0;

  return (
    <div className="flex h-full min-w-0 flex-col gap-4">
      <SegmentedControl
        value={tab}
        onChange={setTab}
        items={items}
        ariaLabel="Insights tabs"
      />

      <div className="min-h-0 min-w-0 flex-1 overflow-auto">
        {tab === "predictions" &&
          (hasPredictions ? (
            <PredictionsPanel
              predictions={predictions}
              currentLap={currentLap}
              modelMetrics={modelMetrics}
              totalLaps={totalLaps}
            />
          ) : (
            <EmptyState
              title="Model is training"
              subtitle="Predictions will appear once enough laps are processed."
            />
          ))}

        {tab === "battles" &&
          (hasDrivers ? (
            <BattlesWidget drivers={drivers} />
          ) : (
            <EmptyState
              title="No battles yet"
              subtitle="Battles will show once gaps tighten."
            />
          ))}

        {tab === "weather" &&
          (weatherData ? (
            <WeatherWidget data={weatherData} />
          ) : (
            <EmptyState
              title="Weather unavailable"
              subtitle="Waiting for telemetry data."
            />
          ))}

        {tab === "notifications" &&
          (hasNotifications ? (
            <NotificationsPanel notifications={notifications} />
          ) : (
            <EmptyState
              title="No events yet"
              subtitle="Race events will appear here."
            />
          ))}
      </div>
    </div>
  );
}