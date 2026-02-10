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
  const [selectedRace, setSelectedRace] = useState(21);

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

        apiClient.on("race/started", () => {
          setRaceRunning(true);
          setNotifications((prev) => [
            {
              id: Date.now(),
              type: "success",
              message: "Race started!",
              time: new Date().toLocaleTimeString(),
            },
            ...prev,
          ]);
        });

        apiClient.on("race/paused", () => {
          setRaceRunning(false);
        });

        apiClient.on("race/resumed", () => {
          setRaceRunning(true);
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
          <p className="text-sm text-neutral-400">
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

      {/* TOOLBAR */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12 lg:items-stretch">
        <div className="lg:col-span-5">
          <Card className="h-full p-4">
            <RaceSelector
              selectedRace={selectedRace}
              onRaceChange={handleRaceChange}
              disabled={raceRunning}
            />
          </Card>
        </div>

        <div className="lg:col-span-7">
          <Card className="h-full p-4">
            <RaceControls
              raceInitialized={raceInitialized}
              raceRunning={raceRunning}
              connected={connected}
              raceData={raceData}
            />
          </Card>
        </div>
      </div>

      {/* WEATHER (FULL WIDTH) */}
      <Card>
        <WeatherWidget data={weatherData} />
      </Card>

      {/* STANDINGS + PREDICTIONS (50/50) */}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <Card className="h-full">
          <DriversList
            drivers={raceData?.drivers || []}
            currentLap={raceData?.currentLap}
          />
        </Card>

        <Card className="h-full">
          <PredictionsPanel
            predictions={predictions}
            currentLap={raceData?.currentLap}
            modelMetrics={modelMetrics}
            totalLaps={raceData?.totalLaps}
          />
        </Card>
      </div>

      {/* NOTIFICATIONS (FULL WIDTH) */}
      <Card>
        <NotificationsPanel notifications={notifications} />
      </Card>

    </div>
  );
}
