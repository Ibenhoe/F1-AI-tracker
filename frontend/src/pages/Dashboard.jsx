import { useState, useEffect } from "react";

import WeatherWidget from "../components/WeatherWidget";
import DriversList from "../components/DriversList";
import PredictionsPanel from "../components/PredictionsPanel";
import NotificationsPanel from "../components/NotificationsPanel";
import RaceSelector from "../components/RaceSelector";
import PositionChart from "../components/PositionChart";
import BattlesWidget from "../components/BattlesWidget";
import apiClient from "../services/apiClient";

import Card from "../components/ui/Card";
import Badge from "../components/ui/Badge";

function InsightsCard({
  predictions,
  currentLap,
  modelMetrics,
  totalLaps,
  weatherData,
  notifications,
  drivers,
}) {

  const [tab, setTab] = useState("predictions");
  const notifCount = Array.isArray(notifications) ? notifications.length : 0;

  const tabs = [
    { id: "predictions", label: "Predictions" },
    { id: "battles", label: "Battles" },
    { id: "weather", label: "Weather" },
    { id: "notifications", label: "Notifications" },
  ];

  return (
    <div className="flex h-full min-w-0 flex-col gap-4">
      <div className="flex h-full min-w-0 flex-col gap-4">
        {/* Tabs / Segmented control */}
        <div className="min-w-0">
          <div
            className={[
              "inline-flex w-full min-w-0 items-center justify-center gap-1",
              "overflow-x-auto",
              "rounded-2xl border border-neutral-200/70 bg-white/60 p-1 backdrop-blur",
              "dark:border-white/10 dark:bg-neutral-950/30",
            ].join(" ")}
            role="tablist"
            aria-label="Insights tabs"
          >
            {tabs.map((t) => {
              const isActive = tab === t.id;
              const showNotifBadge = t.id === "notifications" && notifCount > 0;
              const badgeText = notifCount > 99 ? "99+" : String(notifCount);

              return (
                <button
                  key={t.id}
                  type="button"
                  onClick={() => setTab(t.id)}
                  role="tab"
                  aria-selected={isActive}
                  className={[
                    "group",
                    "relative inline-flex shrink-0 items-center justify-center gap-1.5 text-center",
                    "rounded-lg px-3 py-1 text-sm font-medium",
                    "transition-colors",
                    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[rgb(var(--accent))] focus-visible:ring-offset-2",
                    "focus-visible:ring-offset-white dark:focus-visible:ring-offset-neutral-950",
                    isActive
                      ? "bg-neutral-900 text-white dark:bg-white/10 dark:text-neutral-50"
                      : "text-neutral-600 hover:bg-neutral-100/70 dark:text-neutral-300 dark:hover:bg-white/5",
                  ].join(" ")}
                >
                  <span className="whitespace-nowrap">{t.label}</span>

                  {showNotifBadge ? (
                    <span
                      className={[
                        "inline-flex items-center justify-center",
                        "h-5 min-w-5 px-1.5",
                        "rounded-full",
                        "text-[11px] font-bold tabular-nums leading-none",
                        "bg-[rgb(var(--accent))] text-[rgb(var(--accent-fg))]",
                        "ring-2 ring-white dark:ring-neutral-950",
                        "shadow-[0_10px_22px_rgba(0,0,0,0.22)]",
                      ].join(" ")}
                      aria-label={`${notifCount} notifications`}
                      title={`${notifCount} notifications`}
                    >
                      {badgeText}
                    </span>
                  ) : null}
                </button>
              );
            })}
          </div>
        </div>

        <div className="min-h-0 min-w-0 flex-1 overflow-auto">
          {tab === "predictions" && (
            <PredictionsPanel
              predictions={predictions}
              currentLap={currentLap}
              modelMetrics={modelMetrics}
              totalLaps={totalLaps}
            />
          )}

          {tab === "battles" && <BattlesWidget drivers={drivers} />}

          {tab === "weather" && <WeatherWidget data={weatherData} />}

          {tab === "notifications" && (
            <NotificationsPanel notifications={notifications} />
          )}
        </div>
      </div>
    </div>
  );
}

function StatsBar({ drivers, currentLap, totalLaps, raceRunning, trackStatus }) {
  const sorted = [...(drivers || [])].sort((a, b) => a.position - b.position);
  const leader = sorted[0];

  // Fastest lap: driver with shortest lap_time string (rough heuristic)
  const fastestDriver = sorted.reduce((best, d) => {
    if (!d.lap_time || d.lap_time === "—") return best;
    if (!best) return d;
    return d.lap_time < best.lap_time ? d : best;
  }, null);

  const pct = totalLaps > 0 ? Math.round((currentLap / totalLaps) * 100) : 0;

  // Map FastF1 track status codes to display info
  const FLAG_MAP = {
    '1': {
      label: 'Green Flag',
      sub: 'Track clear',
      color: 'text-emerald-500 dark:text-emerald-400',
      flag: <span className="inline-block w-8 h-5 rounded-sm bg-emerald-500 shadow-[0_0_8px_2px_rgba(16,185,129,0.45)]" />,
    },
    '2': {
      label: 'Yellow Flag',
      sub: 'Hazard on track',
      color: 'text-yellow-500 dark:text-yellow-400',
      flag: <span className="inline-block w-8 h-5 rounded-sm bg-yellow-400 shadow-[0_0_8px_2px_rgba(234,179,8,0.45)]" />,
    },
    '4': {
      label: 'Safety Car',
      sub: 'SC deployed',
      color: 'text-orange-500 dark:text-orange-400',
      flag: (
        <span className="inline-flex items-center justify-center w-8 h-5 rounded-sm bg-orange-500 text-[9px] font-black text-white tracking-tight shadow-[0_0_8px_2px_rgba(249,115,22,0.45)]">
          SC
        </span>
      ),
    },
    '5': {
      label: 'Red Flag',
      sub: 'Race suspended',
      color: 'text-red-500 dark:text-red-400',
      flag: <span className="inline-block w-8 h-5 rounded-sm bg-red-600 shadow-[0_0_8px_2px_rgba(220,38,38,0.55)]" />,
    },
    '6': {
      label: 'Virtual SC',
      sub: 'VSC deployed',
      color: 'text-amber-500 dark:text-amber-400',
      flag: (
        <span className="inline-flex items-center justify-center w-8 h-5 rounded-sm bg-yellow-400 text-[8px] font-black text-neutral-900 tracking-tight shadow-[0_0_8px_2px_rgba(234,179,8,0.45)]">
          VSC
        </span>
      ),
    },
  };
  const flag = FLAG_MAP[String(trackStatus)] || FLAG_MAP['1'];

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {/* Race progress */}
      <Card className="p-4" clip>
        <p className="text-xs text-neutral-500 dark:text-neutral-400 mb-1">Race Progress</p>
        <p className="text-lg font-bold tabular-nums text-neutral-900 dark:text-neutral-50">{currentLap}<span className="text-sm font-normal text-neutral-500"> / {totalLaps} laps</span></p>
        <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-neutral-200 dark:bg-neutral-700">
          <div
            className="h-full rounded-full bg-[rgb(var(--accent))] transition-all duration-700"
            style={{ width: `${pct}%` }}
          />
        </div>
      </Card>

      {/* Race leader */}
      <Card className="p-4" clip>
        <p className="text-xs text-neutral-500 dark:text-neutral-400 mb-1">Race Leader</p>
        {leader ? (
          <>
            <p className="text-lg font-bold text-neutral-900 dark:text-neutral-50">{leader.driver_code}</p>
            <p className="text-xs text-neutral-500 dark:text-neutral-400 truncate">{leader.team}</p>
          </>
        ) : (
          <p className="text-sm text-neutral-400">—</p>
        )}
      </Card>

      {/* Fastest lap */}
      <Card className="p-4" clip>
        <p className="text-xs text-neutral-500 dark:text-neutral-400 mb-1">Fastest Lap</p>
        {fastestDriver ? (
          <>
            <p className="text-lg font-bold text-purple-600 dark:text-purple-400">{fastestDriver.driver_code}</p>
            <p className="text-xs font-mono text-purple-500 dark:text-purple-400">{fastestDriver.lap_time}</p>
          </>
        ) : (
          <p className="text-sm text-neutral-400">—</p>
        )}
      </Card>

      {/* Race Flag */}
      <Card className="p-4" clip>
        <p className="text-xs text-neutral-500 dark:text-neutral-400 mb-1">Race Flag</p>
        <div className="flex items-center gap-2.5 mt-0.5">
          {flag.flag}
          <p className={`text-base font-bold leading-tight ${flag.color}`}>{flag.label}</p>
        </div>
        <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-1">{flag.sub}</p>
      </Card>
    </div>
  );
}

export default function Dashboard() {
  const [raceData, setRaceData] = useState(null);
  const [weatherData, setWeatherData] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [modelMetrics, setModelMetrics] = useState(null);
  const [notifications, setNotifications] = useState([]);
  const [currentLap, setCurrentLap] = useState(0);
  const [raceRunning, setRaceRunning] = useState(false);
  const [connected, setConnected] = useState(false);
  const [raceInitialized, setRaceInitialized] = useState(false);
  const [trackStatus, setTrackStatus] = useState('1');

  // Lap-by-lap position history for the chart
  const [lapHistory, setLapHistory] = useState([]);

  // True only when race/ready event received
  const [raceReady, setRaceReady] = useState(false);

  const [selectedRace, setSelectedRace] = useState(1);

  // Chart: which drivers to highlight (top 5 predicted + leader)
  const [chartFocus, setChartFocus] = useState(null);

  // track loading state for the selector UX
  const [raceLoading, setRaceLoading] = useState(false);

  // true once Start has been pressed at least once; prevents Resume from
  // firing before the race has ever begun, which would spawn a stale thread.
  const [raceEverStarted, setRaceEverStarted] = useState(false);

  const [simSpeed, setSimSpeed] = useState(1.0);

  const [mainPanel, setMainPanel] = useState("standings"); // "standings" | "history"

  const canStart = connected && raceReady && !raceRunning && !raceEverStarted;
  const canPause = connected && raceRunning;
  const canResume = connected && raceReady && !raceRunning && raceEverStarted;

  const handleStart = async () => {
    if (!canStart) return;
    try {
      await apiClient.startRace(simSpeed);
      setRaceRunning(true);
      setRaceEverStarted(true);
    } catch (e) {
      console.error("[DASHBOARD] startRace failed:", e);
    }
  };

  const handlePause = async () => {
    if (!canPause) return;
    try {
      await apiClient.pauseRace();
      setRaceRunning(false);
    } catch (e) {
      console.error("[DASHBOARD] pauseRace failed:", e);
    }
  };

  const handleResume = async () => {
    if (!canResume) return;
    try {
      await apiClient.resumeRace();
      setRaceRunning(true);
    } catch (e) {
      console.error("[DASHBOARD] resumeRace failed:", e);
    }
  };

  const handleSpeedChange = (newSpeed) => {
    setSimSpeed(newSpeed);
    if (connected) apiClient.setSimulationSpeed(newSpeed);
  };

  useEffect(() => {
    // Guard flag: if selectedRace changes while async init is still running,
    // the cleanup sets this to true and we skip any deferred listener registrations
    // from the stale call – preventing duplicate / stale handlers.
    let cancelled = false;

    const initializeApp = async () => {
      try {
        setRaceLoading(true);
        setRaceReady(false);
        setRaceInitialized(false);

        await apiClient.connect();
        if (cancelled) return;
        setConnected(true);
        console.log("[DASHBOARD] Connected to backend");

        const result = await apiClient.initRace(selectedRace);
        if (cancelled) return;
        console.log("[DASHBOARD] Race initialized:", result);

        setRaceData({
          race: result.race_name || `Race ${selectedRace}`,
          totalLaps: result.total_laps || 58,
          currentLap: 0,
          drivers: result.drivers || [],
        });

        if (result.weather) {
          setWeatherData({
            temp: result.weather.air_temp || 25,
            humidity: result.weather.humidity || 50,
            trackTemp: result.weather.track_temp || 35,
            windSpeed: result.weather.wind_speed || 0,
            condition: result.weather.conditions || "Dry",
          });
        }

        setRaceInitialized(true);

        // Initialize model metrics with default values
        setModelMetrics({
          total_updates: 0,
          model_maturity_percentage: 0,
          learning_status: "Initializing",
          model_type: "SGD + MLP + XGBoost",
          confidence_cap: "85%",
          is_pretrained: true,
          samples_processed: 0,
        });

        // Register application-level socket listeners.
        // apiClient.disconnect() in the cleanup already cleared any stale
        // handlers from the previous race, so these are always fresh.
        apiClient.on("race/ready", (data) => {
          console.log("[DASHBOARD] Race ready event received:", data);

          setRaceData({
            race: data.race_name || `Race ${data.race_id}`,
            totalLaps: data.total_laps || 58,
            currentLap: 0,
            drivers: data.drivers || [],
          });

          setRaceReady(true);
          setRaceLoading(false);
        });

        apiClient.on("lap/update", (data) => {
          console.log("[DASHBOARD] Lap update:", data.lap_number);

          setCurrentLap(data.lap_number);

          if (data.track_status) setTrackStatus(data.track_status);

          setRaceData((prev) => ({
            ...prev,
            currentLap: data.lap_number,
            drivers: data.drivers,
          }));

          // Accumulate lap history for position chart
          if (data.drivers && data.drivers.length > 0) {
            setLapHistory((prev) => {
              // avoid duplicates
              if (prev.length > 0 && prev[prev.length - 1].lap === data.lap_number) return prev;
              const frame = {
                lap: data.lap_number,
                drivers: data.drivers.map((d) => ({
                  code: d.driver_code,
                  position: d.position,
                  team: d.team,
                })),
              };
              return [...prev, frame];
            });

            // Update chart focus: highlight predicted top 5 + current leader
            if (data.predictions && data.predictions.length > 0) {
              const focusCodes = data.predictions.slice(0, 5).map((p) => p.driver_code).filter(Boolean);
              setChartFocus(focusCodes.length > 0 ? focusCodes : null);
            }
          }

          setPredictions(data.predictions);

          if (data.model_metrics) {
            setModelMetrics(data.model_metrics);
            console.log("[DASHBOARD] Model metrics updated:", data.model_metrics);
          }

          if (data.weather) {
            setWeatherData({
              temp: data.weather.air_temp || 25,
              humidity: data.weather.humidity || 50,
              trackTemp: data.weather.track_temp || 35,
              windSpeed: data.weather.wind_speed || 0,
              condition: data.weather.conditions || "Dry",
            });
          }

          if (data.events && data.events.length > 0) {
            setNotifications((prev) => {
              const newNotifications = data.events.map((e, i) => ({
                id: `${e.id || Date.now()}-${i}-${Math.random()
                  .toString(36)
                  .substr(2, 9)}`,
                type: e.type || "info",
                color_code: e.color_code || undefined,
                message: e.message,
                time: new Date().toLocaleTimeString(),
                lapNumber: data.lap_number,
              }));

              const allNotifications = [...newNotifications, ...prev];
              const seen = new Set();
              const unique = allNotifications.filter((n) => {
                const key = `${n.lapNumber}-${n.message}`;
                if (seen.has(key)) return false;
                seen.add(key);
                return true;
              });

              return unique.slice(0, 5);
            });
          }
        });

        apiClient.on("race/finished", () => {
          setRaceRunning(false);
          setNotifications((prev) => [
            {
              id: Date.now(),
              type: "success",
              message: "Race finished!",
              time: new Date().toLocaleTimeString(),
            },
            ...prev,
          ]);
        });

        apiClient.on("race/error", (data) => {
          setRaceLoading(false);
          setNotifications((prev) => [
            {
              id: Date.now(),
              type: "error",
              message: data.error,
              time: new Date().toLocaleTimeString(),
            },
            ...prev,
          ]);
        });
      } catch (error) {
        if (cancelled) return;
        console.error("[DASHBOARD] Initialization error:", error);
        setRaceLoading(false);
        setNotifications([
          {
            id: Date.now(),
            type: "error",
            message: `Connection error: ${error.message}`,
            time: new Date().toLocaleTimeString(),
          },
        ]);
      }
    };

    initializeApp();

    return () => {
      // Signal any in-flight async init that it is stale
      cancelled = true;
      // Clear application-level listeners so the next race doesn't inherit them.
      // The underlying socket stays alive (no unnecessary disconnect/reconnect).
      apiClient.disconnect();
    };
  }, [selectedRace]);

  const handleRaceChange = async (newRaceNumber) => {
    setSelectedRace(newRaceNumber);
    setRaceInitialized(false);
    setRaceReady(false);
    setRaceLoading(true);
    setCurrentLap(0);
    setRaceRunning(false);
    setRaceEverStarted(false);
    setTrackStatus('1');
    setLapHistory([]);
    setChartFocus(null);
    setMainPanel("standings");
    setSimSpeed(1.0);
  };

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
          <div className="flex flex-wrap items-center gap-2 rounded-2xl border border-neutral-200/70 bg-white/60 px-3 py-2 backdrop-blur dark:border-white/10 dark:bg-neutral-950/30">
            <Badge variant={connected ? "success" : "danger"}>
              {connected ? "Connected" : "Disconnected"}
            </Badge>
            <Badge variant={raceRunning ? "warning" : "neutral"}>
              {raceRunning ? "Race running" : "Paused"}
            </Badge>
            <Badge variant="accent">Lap {currentLap}</Badge>
          </div>
        </div>
      </div>

      {/* TOP CONTROLS */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <p className="text-[11px] font-semibold uppercase tracking-widest text-neutral-400 dark:text-neutral-600">
            Session
          </p>
        </div>

        <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
          <Card className="lg:col-span-4 p-5 overflow-hidden" clip>
            <div className="h-[320px] overflow-hidden">
              <div className="h-full overflow-auto pr-1">
                <RaceSelector
                  selectedRace={selectedRace}
                  onRaceChange={handleRaceChange}
                  disabled={false}
                  raceLoading={raceLoading}
                  raceReady={raceReady}
                  raceRunning={raceRunning}
                  raceEverStarted={raceEverStarted}
                  speed={simSpeed}
                  onStart={handleStart}
                  onPause={handlePause}
                  onResume={handleResume}
                  onSpeedChange={handleSpeedChange}
                />
              </div>
            </div>
          </Card>

          <Card className="lg:col-span-8 p-5 overflow-hidden" clip>
            <div className="h-[320px] overflow-hidden">
              <div className="h-full overflow-auto pr-1">
                <InsightsCard
                  predictions={predictions}
                  currentLap={raceData?.currentLap}
                  modelMetrics={modelMetrics}
                  totalLaps={raceData?.totalLaps}
                  weatherData={weatherData}
                  notifications={notifications}
                  drivers={raceData?.drivers || []}
                />
              </div>
            </div>
          </Card>
        </div>
      </div>

      {/* STATS BAR */}
      <StatsBar
        drivers={raceData?.drivers || []}
        currentLap={currentLap}
        totalLaps={raceData?.totalLaps || 0}
        raceRunning={raceRunning}
        trackStatus={trackStatus}
      />

      {/* MAIN PANEL */}
      <Card className="p-5" clip>
        <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
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

          <div className="flex flex-wrap items-center gap-2">
            {/* Segmented control */}
            <div className="inline-flex rounded-2xl border border-neutral-200/70 bg-white/60 p-1 backdrop-blur dark:border-white/10 dark:bg-neutral-950/30">
              {[
                { id: "standings", label: "Standings" },
                { id: "history", label: "Position history" },
              ].map((opt) => (
                <button
                  key={opt.id}
                  type="button"
                  onClick={() => setMainPanel(opt.id)}
                  className={[
                    "rounded-xl px-3 py-1.5 text-sm font-medium transition",
                    mainPanel === opt.id
                      ? "bg-neutral-900 text-white dark:bg-white/10 dark:text-neutral-50"
                      : "text-neutral-600 hover:bg-neutral-100/70 dark:text-neutral-300 dark:hover:bg-white/5",
                  ].join(" ")}
                >
                  {opt.label}
                </button>
              ))}
            </div>

            {/* Context badge */}
            {mainPanel === "standings" ? (
              <Badge variant="neutral">Lap {currentLap}</Badge>
            ) : lapHistory.length > 1 ? (
              <Badge variant="neutral">{lapHistory.length} points</Badge>
            ) : null}
          </div>
        </div>

        {/* Content */}
        {mainPanel === "standings" ? (
          <div className="min-w-0">
            <DriversList
              drivers={raceData?.drivers || []}
              currentLap={raceData?.currentLap}
            />
          </div>
        ) : (
          <PositionChart
            lapHistory={lapHistory}
            totalLaps={raceData?.totalLaps || 60}
            highlightedDrivers={chartFocus}
          />
        )}
      </Card>
    </div >
  );
}
