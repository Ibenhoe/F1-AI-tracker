// src/components/wiki/hooks/useWikipediaSummary.js
import { useEffect, useState } from "react";

export default function useWikipediaSummary(title) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!title) {
      setData(null);
      return;
    }

    let cancelled = false;

    setLoading(true);
    setData(null);

    fetch(
      `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(
        title
      )}`
    )
      .then((r) => (r.ok ? r.json() : null))
      .then((json) => {
        if (!cancelled) setData(json);
      })
      .catch(() => {
        if (!cancelled) setData(null);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [title]);

  return { data, loading };
}