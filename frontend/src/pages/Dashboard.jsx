import { useState, useEffect } from "react";

import WeatherWidget from "../components/WeatherWidget";
import DriversList from "../components/DriversList";
import PredictionsPanel from "../components/PredictionsPanel";
import NotificationsPanel from "../components/NotificationsPanel";
import RaceControls from "../components/RaceControls";
import RaceSelector from "../components/RaceSelector";
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
}) {
  const [tab, setTab] = useState("predictions");

  const tabs = [
    { id: "predictions", label: "Predictions" },
    { id: "weather", label: "Weather" },
    { id: "notifications", label: "Notifications" },
  ];

  return (
    <Card className="xl:col-span-4 p-5" clip>
      <div className="flex h-full min-w-0 flex-col gap-4">
        <div className="flex flex-wrap gap-2">
          {tabs.map((t) => (
            <button
              key={t.id}
              type="button"
              onClick={() => setTab(t.id)}
              className={[
                "rounded-full border px-3 py-1 text-sm font-medium transition",
                tab === t.id
                  ? "border-neutral-900 bg-neutral-900 text-white dark:border-neutral-200 dark:bg-neutral-100 dark:text-neutral-900"
                  : "border-neutral-200 bg-white text-neutral-700 hover:bg-neutral-50 dark:border-neutral-800 dark:bg-neutral-950/40 dark:text-neutral-200 dark:hover:bg-neutral-900/40",
              ].join(" ")}
            >
              {t.label}
            </button>
          ))}
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

          {tab === "weather" && <WeatherWidget data={weatherData} />}

          {tab === "notifications" && (
            <NotificationsPanel notifications={notifications} />
          )}
        </div>
      </div>
    </Card>
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
  const [selectedRace, setSelectedRace] = useState(1);

  useEffect(() => {
    const initializeApp = async () => {
      try {
        await apiClient.connect();
        setConnected(true);
        console.log("[DASHBOARD] Connected to backend");

        const result = await apiClient.initRace(selectedRace);
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

        // Listen for race ready event when a new race is initialized
        apiClient.on("race/ready", (data) => {
          console.log("[DASHBOARD] Race ready event received:", data);
          
          setRaceData({
            race: data.race_name || `Race ${data.race_id}`,
            totalLaps: data.total_laps || 58,
            currentLap: 0,
            drivers: data.drivers || [],
          });
        });

        apiClient.on("lap/update", (data) => {
          console.log("[DASHBOARD] Lap update:", data.lap_number);

          setCurrentLap(data.lap_number);

          setRaceData((prev) => ({
            ...prev,
            currentLap: data.lap_number,
            drivers: data.drivers,
          }));

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
        console.error("[DASHBOARD] Initialization error:", error);
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
      apiClient.disconnect();
    };
  }, [selectedRace]);

  const handleRaceChange = async (newRaceNumber) => {
    setSelectedRace(newRaceNumber);
    setRaceInitialized(false);
    setCurrentLap(0);
    setRaceRunning(false);
  };

  return (
    <div className="space-y-6">
      {/* HEADER */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div className="space-y-1">
          <h1 className="text-xl font-semibold tracking-tight">Dashboard</h1>
          <p className="text-sm text-neutral-500 dark:text-neutral-400">
            Live race tracking, model predictions and telemetry summaries.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Badge variant={connected ? "success" : "danger"}>
            {connected ? "Connected" : "Disconnected"}
          </Badge>
          <Badge variant={raceRunning ? "warning" : "neutral"}>
            {raceRunning ? "Race running" : "Paused"}
          </Badge>
          <Badge variant="neutral">Lap {currentLap}</Badge>
        </div>
      </div>

      {/* TOP CONTROLS (compact, calm) */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        <Card className="lg:col-span-4 p-4" clip>
          <RaceSelector
            selectedRace={selectedRace}
            onRaceChange={handleRaceChange}
            disabled={raceRunning}
          />
        </Card>

        <Card className="lg:col-span-8 p-4" clip>
          <RaceControls
            raceInitialized={raceInitialized}
            raceRunning={raceRunning}
            connected={connected}
            raceData={raceData}
            onStarted={() => setRaceRunning(true)}
            onPaused={() => setRaceRunning(false)}
            onResumed={() => setRaceRunning(true)}
          />
        </Card>
      </div>

      {/* MAIN GRID */}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-12">
        {/* PRIMARY */}
        <Card className="xl:col-span-8 p-5" clip>
          <div className="flex min-w-0 flex-col gap-4">
            <div className="min-w-0">
              <DriversList
                drivers={raceData?.drivers || []}
                currentLap={raceData?.currentLap}
              />
            </div>
          </div>
        </Card>

        {/* INSIGHTS */}
        <InsightsCard
          predictions={predictions}
          currentLap={raceData?.currentLap}
          modelMetrics={modelMetrics}
          totalLaps={raceData?.totalLaps}
          weatherData={weatherData}
          notifications={notifications}
        />
      </div>
    </div>
  );
}
