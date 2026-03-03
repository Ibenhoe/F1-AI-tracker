import { useCallback, useEffect, useMemo, useState } from "react";
import apiClient from "../../../services/apiClient";

export default function useDashboardRace({ initialRace = 1 } = {}) {
  const [raceData, setRaceData] = useState(null);
  const [weatherData, setWeatherData] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [modelMetrics, setModelMetrics] = useState(null);
  const [notifications, setNotifications] = useState([]);
  const [currentLap, setCurrentLap] = useState(0);

  const [raceRunning, setRaceRunning] = useState(false);
  const [connected, setConnected] = useState(false);

  const [raceReady, setRaceReady] = useState(false);
  const [raceLoading, setRaceLoading] = useState(false);
  const [raceInitialized, setRaceInitialized] = useState(false);

  const [trackStatus, setTrackStatus] = useState("1");

  const [lapHistory, setLapHistory] = useState([]);
  const [chartFocus, setChartFocus] = useState(null);

  const [selectedRace, setSelectedRace] = useState(initialRace);

  const [raceEverStarted, setRaceEverStarted] = useState(false);
  const [simSpeed, setSimSpeed] = useState(1.0);

  const canStart = connected && raceReady && !raceRunning && !raceEverStarted;
  const canPause = connected && raceRunning;
  const canResume = connected && raceReady && !raceRunning && raceEverStarted;

  const setSpeed = useCallback(
    (newSpeed) => {
      setSimSpeed(newSpeed);
      if (connected) apiClient.setSimulationSpeed(newSpeed);
    },
    [connected]
  );

  const startRace = useCallback(async () => {
    if (!canStart) return;
    try {
      await apiClient.startRace(simSpeed);
      setRaceRunning(true);
      setRaceEverStarted(true);
    } catch (e) {
      console.error("[DASHBOARD] startRace failed:", e);
    }
  }, [canStart, simSpeed]);

  const pauseRace = useCallback(async () => {
    if (!canPause) return;
    try {
      await apiClient.pauseRace();
      setRaceRunning(false);
    } catch (e) {
      console.error("[DASHBOARD] pauseRace failed:", e);
    }
  }, [canPause]);

  const resumeRace = useCallback(async () => {
    if (!canResume) return;
    try {
      await apiClient.resumeRace();
      setRaceRunning(true);
    } catch (e) {
      console.error("[DASHBOARD] resumeRace failed:", e);
    }
  }, [canResume]);

  const changeRace = useCallback((newRaceNumber) => {
    setSelectedRace(newRaceNumber);

    setRaceInitialized(false);
    setRaceReady(false);
    setRaceLoading(true);

    setCurrentLap(0);
    setRaceRunning(false);
    setRaceEverStarted(false);

    setTrackStatus("1");
    setLapHistory([]);
    setChartFocus(null);

    setSimSpeed(1.0);
    setPredictions([]);
    setNotifications([]);
  }, []);

  useEffect(() => {
    let cancelled = false;

    // ---- Define handlers ONCE per effect run so we can off() them in cleanup
    const onRaceReady = (data) => {
      setRaceData({
        race: data.race_name || `Race ${data.race_id}`,
        totalLaps: data.total_laps || 58,
        currentLap: 0,
        drivers: data.drivers || [],
      });

      setRaceReady(true);
      setRaceLoading(false);
    };

    const onLapUpdate = (data) => {
      setCurrentLap(data.lap_number);

      if (data.track_status) setTrackStatus(data.track_status);

      setRaceData((prev) => ({
        ...prev,
        currentLap: data.lap_number,
        drivers: data.drivers,
      }));

      if (Array.isArray(data.drivers) && data.drivers.length > 0) {
        setLapHistory((prev) => {
          if (prev.length > 0 && prev[prev.length - 1].lap === data.lap_number) return prev;

          return [
            ...prev,
            {
              lap: data.lap_number,
              drivers: data.drivers.map((d) => ({
                code: d.driver_code,
                position: d.position,
                team: d.team,
              })),
            },
          ];
        });

        if (Array.isArray(data.predictions) && data.predictions.length > 0) {
          const focusCodes = data.predictions
            .slice(0, 5)
            .map((p) => p.driver_code)
            .filter(Boolean);

          setChartFocus(focusCodes.length > 0 ? focusCodes : null);
        }
      }

      setPredictions(data.predictions || []);

      if (data.model_metrics) setModelMetrics(data.model_metrics);

      if (data.weather) {
        setWeatherData({
          temp: data.weather.air_temp || 25,
          humidity: data.weather.humidity || 50,
          trackTemp: data.weather.track_temp || 35,
          windSpeed: data.weather.wind_speed || 0,
          condition: data.weather.conditions || "Dry",
        });
      }

      if (Array.isArray(data.events) && data.events.length > 0) {
        setNotifications((prev) => {
          const newNotifications = data.events.map((e, i) => ({
            id: `${e.id || Date.now()}-${i}-${Math.random().toString(36).substr(2, 9)}`,
            type: e.type || "info",
            color_code: e.color_code || undefined,
            message: e.message,
            time: new Date().toLocaleTimeString(),
            lapNumber: data.lap_number,
          }));

          const all = [...newNotifications, ...prev];

          const seen = new Set();
          const unique = all.filter((n) => {
            const key = `${n.lapNumber}-${n.message}`;
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
          });

          return unique.slice(0, 5);
        });
      }
    };

    const onRaceFinished = () => {
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
    };

    const onRaceError = (data) => {
      setRaceLoading(false);
      setNotifications((prev) => [
        {
          id: Date.now(),
          type: "error",
          message: data?.error || "Race error",
          time: new Date().toLocaleTimeString(),
        },
        ...prev,
      ]);
    };

    const init = async () => {
      try {
        setRaceLoading(true);
        setRaceReady(false);
        setRaceInitialized(false);

        await apiClient.connect();
        if (cancelled) return;

        setConnected(true);

        // IMPORTANT: register listeners BEFORE initRace response starts arriving
        apiClient.on("race/ready", onRaceReady);
        apiClient.on("lap/update", onLapUpdate);
        apiClient.on("race/finished", onRaceFinished);
        apiClient.on("race/error", onRaceError);

        const result = await apiClient.initRace(selectedRace);
        if (cancelled) return;

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
        } else {
          setWeatherData(null);
        }

        setRaceInitialized(true);

        setModelMetrics({
          total_updates: 0,
          model_maturity_percentage: 0,
          learning_status: "Initializing",
          model_type: "SGD + MLP + XGBoost",
          confidence_cap: "85%",
          is_pretrained: true,
          samples_processed: 0,
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

    init();

    return () => {
      cancelled = true;
      
      apiClient.off("race/ready", onRaceReady);
      apiClient.off("lap/update", onLapUpdate);
      apiClient.off("race/finished", onRaceFinished);
      apiClient.off("race/error", onRaceError);
    };
  }, [selectedRace]);

  const actions = useMemo(
    () => ({
      startRace,
      pauseRace,
      resumeRace,
      setSpeed,
      changeRace,
    }),
    [startRace, pauseRace, resumeRace, setSpeed, changeRace]
  );

  const state = useMemo(
    () => ({
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
      raceInitialized,
      trackStatus,
      lapHistory,
      chartFocus,
      selectedRace,
      raceEverStarted,
      simSpeed,
      canStart,
      canPause,
      canResume,
    }),
    [
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
      raceInitialized,
      trackStatus,
      lapHistory,
      chartFocus,
      selectedRace,
      raceEverStarted,
      simSpeed,
      canStart,
      canPause,
      canResume,
    ]
  );

  return { state, actions };
}