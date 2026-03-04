// src/components/racereplay/hooks/useLapPredictions.js
import { useEffect, useState } from "react";

export default function useLapPredictions({ raceNum }) {
  const [lapPredictions, setLapPredictions] = useState(null); // null=loading, {}=loaded-empty
  const [predictionsLoading, setPredictionsLoading] = useState(false);

  useEffect(() => {
    const fetchMLPredictions = async () => {
      try {
        setPredictionsLoading(true);
        const res = await fetch(`/api/race/lap-predictions?race=${raceNum}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        if (data.predictions_by_lap) setLapPredictions(data.predictions_by_lap);
        else setLapPredictions({});
      } catch (err) {
        console.warn("[RaceReplay] ML predictions unavailable, using heuristic fallback:", err.message);
        setLapPredictions({});
      } finally {
        setPredictionsLoading(false);
      }
    };

    fetchMLPredictions();
  }, [raceNum]);

  return { lapPredictions, predictionsLoading };
}