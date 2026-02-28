// src/components/wiki/hooks/useWikiSessionData.js
import { useEffect, useState } from "react";

export default function useWikiSessionData(selectedRaceId, viewType) {
  const [tableData, setTableData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!selectedRaceId) return;

    let cancelled = false;

    async function fetchData() {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch(
          `http://localhost:5000/api/wiki/${selectedRaceId}/${viewType}`
        );
        if (!response.ok) throw new Error("Failed to load data");
        const data = await response.json();
        if (cancelled) return;

        setTableData(Array.isArray(data) ? data : []);
      } catch (err) {
        if (cancelled) return;
        setError("Could not load data.");
        setTableData([]);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    fetchData();
    return () => {
      cancelled = true;
    };
  }, [selectedRaceId, viewType]);

  return { tableData, loading, error };
}