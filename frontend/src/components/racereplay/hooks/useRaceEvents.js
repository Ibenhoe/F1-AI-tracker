// src/components/racereplay/hooks/useRaceEvents.js
import { useEffect, useRef, useState } from "react";

export default function useRaceEvents({ currentFrame, frameIdx }) {
  const [raceEvents, setRaceEvents] = useState([]);

  const lastPitStopStateRef = useRef({});
  const smoothedGapsRef = useRef({});
  const lastProcessedFrameRef = useRef(0);

  useEffect(() => {
    if (!currentFrame || !currentFrame.drivers) return;

    const jumpSize = Math.abs(frameIdx - lastProcessedFrameRef.current);
    if (jumpSize > 5) {
      setRaceEvents([]);
      lastPitStopStateRef.current = {};
    }
    lastProcessedFrameRef.current = frameIdx;

    const newEvents = [];

    Object.entries(currentFrame.drivers).forEach(([code, driver]) => {
      const lastState = lastPitStopStateRef.current[code] || { pit_stops: 0, status: "Running" };

      if (driver.pit_stops > lastState.pit_stops) {
        newEvents.push({
          id: `${currentFrame.lap}-${code}-pit`,
          type: "pit_stop",
          message: `${code} pitted (Stop ${driver.pit_stops})`,
          lap: currentFrame.lap,
          driverCode: code,
          driverName: driver.driver_name || code,
          timestamp: new Date().toLocaleTimeString(),
        });
      }

      if (driver.status === "OUT" && lastState.status !== "OUT") {
        newEvents.push({
          id: `${currentFrame.lap}-${code}-ret`,
          type: "retirement",
          message: `${code} retired`,
          lap: currentFrame.lap,
          driverCode: code,
          driverName: driver.driver_name || code,
          timestamp: new Date().toLocaleTimeString(),
        });
      }

      // smoothed gaps (used by TrackRenderer)
      const currentGapStr = driver.gap || "+0.000";
      const currentGap = parseFloat(currentGapStr.replace("+", "")) || 0;
      const prevSmooth = smoothedGapsRef.current[code] || currentGap;

      let smoothedGap = currentGap;
      if (Math.abs(currentGap - prevSmooth) > 2.0) smoothedGap = prevSmooth + (currentGap - prevSmooth) * 0.1;

      smoothedGapsRef.current[code] = smoothedGap;

      lastPitStopStateRef.current[code] = {
        pit_stops: driver.pit_stops,
        status: driver.status,
      };
    });

    if (newEvents.length > 0) {
      setRaceEvents((prev) => [...newEvents, ...prev].slice(0, 10));
    }
  }, [currentFrame, frameIdx]);

  return { raceEvents, smoothedGapsRef };
}