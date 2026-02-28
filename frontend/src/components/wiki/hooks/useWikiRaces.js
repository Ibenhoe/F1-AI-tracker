// src/components/wiki/hooks/useWikiRaces.js
import { useEffect, useState } from "react";

export default function useWikiRaces(selectedYear) {
  const [races, setRaces] = useState([]);
  const [error, setError] = useState(null);
  const [defaultRaceId, setDefaultRaceId] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function fetchRaces() {
      try {
        const response = await fetch(
          `http://localhost:5000/api/races/${selectedYear}`
        );
        if (!response.ok) {
          throw new Error(`Server antwoordde met status: ${response.status}`);
        }
        const data = await response.json();
        if (cancelled) return;

        setRaces(Array.isArray(data) ? data : []);
        setError(null);

        if (Array.isArray(data) && data.length > 0) {
          setDefaultRaceId(data[0].raceId);
        } else {
          setDefaultRaceId("");
        }
      } catch (err) {
        if (cancelled) return;
        console.error("Failed to fetch races", err);
        setError(
          `Fout bij laden races: ${err.message}. Controleer of de backend draait en de routes heeft.`
        );
        setRaces([]);
        setDefaultRaceId("");
      }
    }

    fetchRaces();
    return () => {
      cancelled = true;
    };
  }, [selectedYear]);

  return { races, error, defaultRaceId };
}