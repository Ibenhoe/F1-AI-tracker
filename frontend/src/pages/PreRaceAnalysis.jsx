// src/pages/PreRaceAnalysis.jsx
import { useEffect, useMemo, useRef, useState } from "react";

import RaceSelector from "../components/dashboard/RaceSelector";
import Card from "../components/ui/Card";
import Badge from "../components/ui/Badge";

import PredictedPodium from "../components/prerace/PredictedPodium";
import PreRaceStatTiles from "../components/prerace/PreRaceStatTiles";
import PreRacePredictionsList from "../components/prerace/PreRacePredictionsList";
import TireStrategyPanel from "../components/prerace/TireStrategyPanel";

import SegmentedControl from "../components/dashboard/components/SegmentedControl";
import EmptyState from "../components/dashboard/components/EmptyState";

/* ---------- small utilities ---------- */
function clamp(n, a, b) {
  const x = Number(n);
  if (!Number.isFinite(x)) return a;
  return Math.max(a, Math.min(b, x));
}

function ErrorCard({ message }) {
  return (
    <Card className="p-4" clip>
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">
            Something went wrong
          </div>
          <div className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">
            {message}
          </div>
        </div>
        <Badge variant="danger">Error</Badge>
      </div>
    </Card>
  );
}

/* ============================= PAGE ============================= */
export default function PreRaceAnalysis() {
  const [raceNumber, setRaceNumber] = useState(1);

  const [predictions, setPredictions] = useState([]);
  const [analysis, setAnalysis] = useState(null);

  const [tireStrategies, setTireStrategies] = useState([]);
  const [circuitAnalysis, setCircuitAnalysis] = useState(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const retryTimerRef = useRef(null);

  const [insightTab, setInsightTab] = useState("podium"); // podium | strategy

  const hasPredictions = Array.isArray(predictions) && predictions.length > 0;
  const top10 = useMemo(() => predictions.slice(0, 10), [predictions]);

  const confStats = useMemo(() => {
    const vals = top10.map((p) => Number(p.confidence)).filter(Number.isFinite);
    const min = vals.length ? Math.min(...vals) : 0;
    const max = vals.length ? Math.max(...vals) : 100;
    const pad = 6;
    const lo = clamp(min - pad, 0, 100);
    const hi = clamp(max + pad, 0, 100);
    return { lo, hi };
  }, [top10]);

  const movers = useMemo(() => {
    const withDelta = top10.map((p, i) => ({
      ...p,
      delta:
        typeof p.grid_position === "number" ? p.grid_position - (i + 1) : 0,
    }));

    const riser = [...withDelta].sort((a, b) => b.delta - a.delta)[0];
    const faller = [...withDelta].sort((a, b) => a.delta - b.delta)[0];
    const topConf = [...withDelta].sort(
      (a, b) => Number(b.confidence ?? 0) - Number(a.confidence ?? 0)
    )[0];

    return { riser, faller, topConf };
  }, [top10]);

  useEffect(() => {
    if (!raceNumber) return;

    fetchPreRaceData(raceNumber);

    return () => {
      if (retryTimerRef.current) clearTimeout(retryTimerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [raceNumber]);

  const fetchPreRaceData = async (raceNum, isPolling = false) => {
    try {
      if (!isPolling) {
        setLoading(true);
        setError(null);
      }

      const [predResponse, tireResponse] = await Promise.all([
        fetch("http://localhost:5000/api/race/prerace-analysis", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ race_number: raceNum, grid: null }),
        }),
        fetch("http://localhost:5000/api/race/tire-strategy", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ race_number: raceNum }),
        }),
      ]);

      if (!predResponse.ok) throw new Error("Failed to fetch pre-race analysis");

      const predData = await predResponse.json();
      if (predData?.predictions) {
        setPredictions(predData.predictions);
        setAnalysis(predData.analysis ?? null);
      } else {
        setPredictions([]);
        setAnalysis(null);
      }

      if (tireResponse.ok || tireResponse.status === 202) {
        const tireData = await tireResponse.json();

        if (tireData?.strategies?.length > 0) {
          setTireStrategies(tireData.strategies);
          setCircuitAnalysis(tireData.circuit_analysis ?? null);
        } else {
          setTireStrategies([]);
          setCircuitAnalysis(null);
        }

        if (tireResponse.status === 202) {
          if (retryTimerRef.current) clearTimeout(retryTimerRef.current);
          retryTimerRef.current = setTimeout(() => {
            fetchPreRaceData(raceNum, true);
          }, 2000);
        }
      } else if (tireResponse.status === 503) {
        setTireStrategies([]);
        setCircuitAnalysis(null);
      }
    } catch (err) {
      console.error("Error fetching pre-race data:", err);
      setError(err?.message ?? "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  const handleRaceSelect = (raceId) => setRaceNumber(raceId);

  return (
    <div className="space-y-6">
      {/* HEADER (EXACT Dashboard style) */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="space-y-1">
          <h1 className="text-2xl font-semibold tracking-tight text-neutral-900 dark:text-neutral-50">
            Pre-race Analysis
          </h1>
          <p className="text-sm text-neutral-500 dark:text-neutral-400">
            AI predictions and tire strategy per race weekend.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <div
            className={[
              "flex flex-wrap items-center gap-2 rounded-2xl px-3 py-2",
              "bg-white dark:bg-neutral-950/40",
              "ring-1 ring-black/5 dark:ring-white/10",
            ].join(" ")}
          >
            <Badge variant="accent">Race {raceNumber}</Badge>
            {loading ? <Badge variant="warning">Loading</Badge> : null}
            {error ? <Badge variant="danger">Error</Badge> : null}
          </div>
        </div>
      </div>

      {/* TOP GRID (EXACT Dashboard structure) */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        {/* LEFT */}
        <Card className="lg:col-span-4 p-5" clip>
          <div className="h-[320px] min-h-0">
            <RaceSelector
              mode="prerace"
              visibleRows={6}
              selectedRace={raceNumber}
              onSelectRace={handleRaceSelect}
              raceLoading={loading}
              raceReady={!loading}
              raceRunning={false}
              raceEverStarted={false}
              speed={1}
            />
          </div>
        </Card>

        {/* RIGHT */}
        <Card className="lg:col-span-8 p-5" clip>
          <div className="flex h-[320px] min-h-0 flex-col gap-4">
            {/* Tabs: same placement style as Dashboard InsightsCard */}
            <SegmentedControl
              value={insightTab}
              onChange={setInsightTab}
              ariaLabel="Pre-race insights tabs"
              items={[
                { id: "podium", label: "Podium" },
                { id: "strategy", label: "Tire strategy" },
              ]}
            />

            <div className="min-h-0 flex-1 overflow-auto">
              {error ? (
                <EmptyState
                  title="Error"
                  subtitle="Fix the backend request and retry."
                />
              ) : loading ? (
                <EmptyState title="Loading" subtitle="Computing insights…" />
              ) : hasPredictions ? (
                insightTab === "podium" ? (
                  <PredictedPodium predictions={predictions} analysis={analysis} />
                ) : (
                  <TireStrategyPanel
                    tireStrategies={tireStrategies}
                    circuitAnalysis={circuitAnalysis}
                  />
                )
              ) : (
                <EmptyState
                  title="No data yet"
                  subtitle="Select a race to generate pre-race predictions."
                />
              )}
            </div>
          </div>
        </Card>
      </div>

      {error ? <ErrorCard message={error} /> : null}

      {/* STATS TILES */}
      <PreRaceStatTiles movers={movers} circuitAnalysis={circuitAnalysis} />

      {/* PREDICTIONS TABLE/LIST */}
      <PreRacePredictionsList predictions={predictions} />
    </div>
  );
}