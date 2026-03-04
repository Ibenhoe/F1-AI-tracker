// src/components/racereplay/hooks/useRealTimeSync.js
import { useEffect } from "react";

export default function useRealTimeSync({ realTimeMode, raceData, setFrameIndex, setRealTimeMode, syncStartRef }) {
  useEffect(() => {
    if (!realTimeMode || !raceData) return;

    const FRAMES_PER_LAP = 120;
    const SECONDS_PER_LAP = 90;

    const interval = setInterval(() => {
      if (!syncStartRef.current) return;

      const elapsedSec = (performance.now() - syncStartRef.current.wallMs) / 1000;
      const targetRaceTime = syncStartRef.current.raceTime + elapsedSec;

      const targetIdx = Math.round((targetRaceTime * FRAMES_PER_LAP) / SECONDS_PER_LAP);
      const clampedIdx = Math.min(Math.max(0, targetIdx), raceData.frames.length - 1);

      setFrameIndex(clampedIdx);

      if (clampedIdx >= raceData.frames.length - 1) {
        setRealTimeMode(false);
        syncStartRef.current = null;
      }
    }, 250);

    return () => clearInterval(interval);
  }, [realTimeMode, raceData, setFrameIndex, setRealTimeMode, syncStartRef]);
}