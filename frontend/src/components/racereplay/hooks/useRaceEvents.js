// src/components/racereplay/hooks/useRaceEvents.js
import { useEffect, useRef, useState } from "react";

export default function useRaceEvents({ currentFrame, frameIdx }) {
  const [raceEvents, setRaceEvents] = useState([]);

  // Per-driver state tracked at lap granularity
  // { [code]: { tire_age, tire_compound, status, pitCount } }
  const driverLapStateRef = useRef({});
  const smoothedGapsRef = useRef({});
  const lastProcessedLapRef = useRef(null);
  const lastProcessedFrameRef = useRef(0);

  useEffect(() => {
    if (!currentFrame || !currentFrame.drivers) return;

    const currentLap = currentFrame.lap;

    // ── Reset on large seek jumps (user scrubbed the timeline) ──────────────
    const jumpSize = Math.abs(frameIdx - lastProcessedFrameRef.current);
    if (jumpSize > 30) {
      setRaceEvents([]);
      driverLapStateRef.current = {};
      lastProcessedLapRef.current = null;
    }
    lastProcessedFrameRef.current = frameIdx;

    // ── Always update smoothed gaps (every frame) ────────────────────────────
    Object.entries(currentFrame.drivers).forEach(([code, driver]) => {
      const currentGapStr = driver.gap || "+0.000";
      const currentGap = parseFloat(currentGapStr.replace("+", "")) || 0;
      const prevSmooth = smoothedGapsRef.current[code] ?? currentGap;
      smoothedGapsRef.current[code] =
        Math.abs(currentGap - prevSmooth) > 2.0
          ? prevSmooth + (currentGap - prevSmooth) * 0.1
          : currentGap;
    });

    // ── Only fire events once per lap change ────────────────────────────────
    if (currentLap === lastProcessedLapRef.current) return;
    lastProcessedLapRef.current = currentLap;

    const newEvents = [];

    Object.entries(currentFrame.drivers).forEach(([code, driver]) => {
      const prev = driverLapStateRef.current[code];

      if (prev) {
        // ── Pit stop detection ───────────────────────────────────────────────
        // pit_stops field is often 0 in cached data, so detect via tire_age:
        // a pit stop resets tire_age — it drops instead of incrementing by 1.
        const tireAgeDropped = driver.tire_age < prev.tire_age;
        const compoundChanged = driver.tire_compound !== prev.tire_compound;
        const pitStopsIncreased = (driver.pit_stops ?? 0) > (prev.pitCount ?? 0);

        if (pitStopsIncreased || tireAgeDropped || compoundChanged) {
          const stopNum = (prev.pitCount ?? 0) + 1;
          newEvents.push({
            id: `${currentLap}-${code}-pit`,
            type: "pit_stop",
            message: `${code} pitted (Stop ${stopNum})`,
            lap: currentLap,
            driverCode: code,
            driverName: driver.driver_name || code,
          });
          driverLapStateRef.current[code] = {
            tire_age: driver.tire_age,
            tire_compound: driver.tire_compound,
            status: driver.status,
            pitCount: stopNum,
          };
          return; // skip status check this lap
        }

        // ── Retirement detection ─────────────────────────────────────────────
        if (driver.status === "OUT" && prev.status !== "OUT") {
          newEvents.push({
            id: `${currentLap}-${code}-ret`,
            type: "retirement",
            message: `${code} retired`,
            lap: currentLap,
            driverCode: code,
            driverName: driver.driver_name || code,
          });
        }
      }

      // Update state for this driver
      driverLapStateRef.current[code] = {
        tire_age: driver.tire_age,
        tire_compound: driver.tire_compound,
        status: driver.status,
        pitCount: prev?.pitCount ?? 0,
      };
    });

    if (newEvents.length > 0) {
      setRaceEvents((prev) => [...newEvents, ...prev].slice(0, 20));
    }
  }, [currentFrame, frameIdx]);

  return { raceEvents, smoothedGapsRef };
}