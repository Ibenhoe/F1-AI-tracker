import { useMemo, useState } from "react";

import {
  RaceSelector,
  DriversList,
  InsightsCard,
  StatsBar,
  SegmentedControl,
  PositionChart
} from "../components/dashboard";

import Card from "../components/ui/Card";
import Badge from "../components/ui/Badge";

import useDashboardRace from "../components/dashboard/hooks/useDashboardRace";

export default function Dashboard() {
  const { state, actions } = useDashboardRace({ initialRace: 1 });

  const {
    raceData,
    weatherData,
    predictions,
    modelMetrics,
    notifications,
    currentLap,
    raceRunning,
    connected,
    raceReady,
    raceLoading,
    trackStatus,
    lapHistory,
    chartFocus,
    selectedRace,
    raceEverStarted,
    simSpeed,
  } = state;

  const { startRace, pauseRace, resumeRace, setSpeed, changeRace } = actions;

  const [mainPanel, setMainPanel] = useState("standings"); // "standings" | "history"

  const mainPanelItems = useMemo(
    () => [
      { id: "standings", label: "Standings" },
      { id: "history", label: "Position history" },
    ],
    []
  );

  return (
    <div className="space-y-6">
      {/* HEADER */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="space-y-1">
          <h1 className="text-2xl font-semibold tracking-tight text-neutral-900 dark:text-neutral-50">
            Dashboard
          </h1>
          <p className="text-sm text-neutral-500 dark:text-neutral-400">
            Live race tracking, model predictions and telemetry summaries.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <div
            className={[
              "flex flex-wrap items-center gap-2 rounded-2xl px-3 py-2",
              "bg-white dark:bg-neutral-950/40",
            ].join(" ")}
          >
            <Badge variant={connected ? "success" : "danger"}>
              {connected ? "Connected" : "Disconnected"}
            </Badge>

            <Badge variant={raceLoading ? "warning" : raceRunning ? "warning" : "neutral"}>
              {raceLoading ? "Loading" : raceRunning ? "Race running" : "Paused"}
            </Badge>
          </div>
        </div>
      </div>

      {/* TOP CONTROLS */}
      <div className="space-y-3">
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
          <Card className="lg:col-span-4 p-5" clip>
            <div className="h-[320px] min-h-0">
              <RaceSelector
                mode="dashboard"
                selectedRace={selectedRace}
                onRaceChange={(n) => {
                  setMainPanel("standings");
                  changeRace(n);
                }}
                disabled={false}
                raceLoading={raceLoading}
                raceReady={raceReady}
                raceRunning={raceRunning}
                raceEverStarted={raceEverStarted}
                speed={simSpeed}
                onStart={startRace}
                onPause={pauseRace}
                onResume={resumeRace}
                onSpeedChange={setSpeed}
              />
            </div>
          </Card>

          <Card className="lg:col-span-8 p-5" clip>
            <div className="h-[320px] min-h-0">
              <InsightsCard
                selectedRace={selectedRace}
                predictions={predictions}
                currentLap={raceData?.currentLap}
                modelMetrics={modelMetrics}
                totalLaps={raceData?.totalLaps}
                weatherData={weatherData}
                notifications={notifications}
                drivers={raceData?.drivers || []}
              />
            </div>
          </Card>
        </div>
      </div>

      {/* STATS BAR */}
      <StatsBar
        drivers={raceData?.drivers || []}
        currentLap={currentLap}
        totalLaps={raceData?.totalLaps || 0}
        trackStatus={trackStatus}
      />

      {/* MAIN PANEL */}
      <Card className="p-5" clip>
        <div className="mb-4 flex flex-col gap-3 sm:grid sm:grid-cols-3 sm:items-end">
          <div className="space-y-1">
            <p className="text-[11px] font-semibold uppercase tracking-widest text-neutral-400 dark:text-neutral-600">
              Live
            </p>

            <h2 className="text-base font-semibold tracking-tight text-neutral-900 dark:text-neutral-50">
              {mainPanel === "standings" ? "Standings" : "Position History"}
            </h2>

            <p className="text-xs text-neutral-500 dark:text-neutral-400">
              {mainPanel === "standings"
                ? "Current classification, last lap and key race indicators."
                : "Driver positions across all laps — AI top-5 highlighted."}
            </p>
          </div>

          <div className="sm:flex sm:justify-center">
            <SegmentedControl
              value={mainPanel}
              onChange={setMainPanel}
              ariaLabel="Main panel"
              items={mainPanelItems}
            />
          </div>
        </div>

        {mainPanel === "standings" ? (
          <div className="min-w-0">
            <DriversList drivers={raceData?.drivers || []} />
          </div>
        ) : (
          <PositionChart
            lapHistory={lapHistory}
            totalLaps={raceData?.totalLaps || 60}
            highlightedDrivers={chartFocus}
          />
        )}
      </Card>
    </div>
  );
}