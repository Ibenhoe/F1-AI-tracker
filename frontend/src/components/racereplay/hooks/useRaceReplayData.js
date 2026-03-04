// src/components/racereplay/hooks/useRaceReplayData.js
import { useEffect, useState } from "react";

export default function useRaceReplayData({ raceNum }) {
  const [raceData, setRaceData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadRaceData = async () => {
      try {
        setLoading(true);

        const response = await fetch(`/api/race/replay-data?race=${raceNum}`);
        if (!response.ok) throw new Error(`Failed to load race data: ${response.status}`);

        const data = await response.json();
        setRaceData(data);
        setError(null);
      } catch (err) {
        setError(err.message);
        console.error("Error loading race data:", err);
      } finally {
        setLoading(false);
      }
    };

    loadRaceData();
  }, [raceNum]);

  return { raceData, loading, error };
}