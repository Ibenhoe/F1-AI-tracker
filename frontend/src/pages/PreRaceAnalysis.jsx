import "./PreRaceAnalysis.css";
import { useState, useEffect, useRef, useMemo } from "react";

import RaceSelector from "../components/RaceSelector";
import Card from "../components/ui/Card";
import Badge from "../components/ui/Badge";

export default function PreRaceAnalysis() {
  const [raceNumber, setRaceNumber] = useState(21);
  const [predictions, setPredictions] = useState([]);
  const [tireStrategies, setTireStrategies] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [circuitAnalysis, setCircuitAnalysis] = useState(null);
  const retryTimerRef = useRef(null);

  const hasPredictions = predictions.length > 0;

  useEffect(() => {
    if (raceNumber) fetchPreRaceData(raceNumber);

    return () => {
      if (retryTimerRef.current) clearTimeout(retryTimerRef.current);
    };
  }, [raceNumber]);

  const fetchPreRaceData = async (raceNum) => {
    try {
      setLoading(true);
      setError(null);

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

      if (!predResponse.ok) {
        throw new Error("Failed to fetch pre-race analysis");
      }

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
            fetchPreRaceData(raceNum);
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

  const top10 = useMemo(() => predictions.slice(0, 10), [predictions]);

  const clamp = (n, a, b) => Math.max(a, Math.min(b, n));

  const confValues = top10
    .map((p) => Number(p.confidence))
    .filter((n) => Number.isFinite(n));

  const confMin = confValues.length ? Math.min(...confValues) : 0;
  const confMax = confValues.length ? Math.max(...confValues) : 100;

  const pad = 6;
  const lo = clamp(confMin - pad, 0, 100);
  const hi = clamp(confMax + pad, 0, 100);

  const confidenceColor = (pct) => {
    const p = clamp(Number(pct ?? 0), 0, 100);

    const t = hi === lo ? 0.5 : clamp((p - lo) / (hi - lo), 0, 1);

    const hue = 15 + t * 85;
    const sat = 85;
    const lit = 52;

    return `hsl(${hue} ${sat}% ${lit}%)`;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div className="space-y-1">
          <h1 className="text-xl font-semibold tracking-tight">Pre-race analysis</h1>
          <p className="text-sm text-neutral-500 dark:text-neutral-400">
            AI predictions and tire strategy per race weekend.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Badge variant="accent">Race {raceNumber}</Badge>
          {loading ? <Badge variant="warning">Loading</Badge> : null}
          {error ? <Badge variant="danger">Error</Badge> : null}
        </div>
      </div>

      {/* Race selector */}
      <Card className="p-4" clip>
        <div className="flex flex-col gap-3">
          <div className="text-xs text-neutral-500 dark:text-neutral-400">
            Select a race to generate pre-race predictions.
          </div>

          <RaceSelector
            selectedRace={raceNumber}
            onSelectRace={handleRaceSelect}
            variant="subtle"
          />
        </div>
      </Card>

      {/* Error */}
      {error ? (
        <Card className="p-4" clip>
          <div className="flex items-start justify-between gap-3">
            <div>
              <div className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">
                Something went wrong
              </div>
              <div className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">
                {error}
              </div>
            </div>
            <Badge variant="danger">Error</Badge>
          </div>
        </Card>
      ) : null}

      {/* Loading */}
      {loading ? (
        <Card className="p-6" clip>
          <div className="flex items-center gap-4">
            <div className="h-10 w-10 rounded-full border-4 border-neutral-200 border-t-neutral-900 animate-spin dark:border-neutral-800 dark:border-t-neutral-100" />
            <div>
              <div className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">
                Generating analysis
              </div>
              <div className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">
                Loading models and computing predictions…
              </div>
            </div>
          </div>
        </Card>
      ) : null}

      {/* Main content */}
      {!loading && hasPredictions ? (
        <div className="grid grid-cols-1 gap-4 xl:grid-cols-12">
          {/* Predictions */}
          <Card className="xl:col-span-8 p-5" clip>
            <div className="flex flex-col gap-4">
              <div>
                <h2 className="text-sm font-semibold tracking-tight">AI predictions</h2>
                <p className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">
                  Top 10 expected finishing order
                </p>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-neutral-200/80 dark:border-white/10">
                      <th className="py-3 pr-4 text-left text-xs font-semibold text-neutral-600 dark:text-neutral-400">
                        #
                      </th>
                      <th className="py-3 pr-4 text-left text-xs font-semibold text-neutral-600 dark:text-neutral-400">
                        Driver
                      </th>
                      <th className="py-3 pr-4 text-left text-xs font-semibold text-neutral-600 dark:text-neutral-400">
                        Team
                      </th>
                      <th className="py-3 pr-4 text-left text-xs font-semibold text-neutral-600 dark:text-neutral-400">
                        Grid
                      </th>
                      <th className="py-3 pr-4 text-left text-xs font-semibold text-neutral-600 dark:text-neutral-400">
                        Score
                      </th>
                      <th className="py-3 text-left text-xs font-semibold text-neutral-600 dark:text-neutral-400">
                        Confidence
                      </th>
                    </tr>
                  </thead>

                  <tbody>
                    {top10.map((pred, idx) => {
                      const driver = pred.driver_name || pred.driver || "Unknown";
                      const number = pred.number ?? "—";
                      const team = pred.team ?? "—";
                      const grid = pred.grid_position ?? "—";
                      const score =
                        typeof pred.ai_score === "number"
                          ? pred.ai_score.toFixed(1)
                          : "—";
                      const conf =
                        typeof pred.confidence === "number"
                          ? Math.max(0, Math.min(100, pred.confidence))
                          : 0;

                      return (
                        <tr
                          key={idx}
                          className="border-b border-neutral-100 hover:bg-neutral-50/70 dark:border-white/5 dark:hover:bg-white/5"
                        >
                          <td className="py-3 pr-4 font-semibold text-neutral-900 dark:text-neutral-100 tabular-nums">
                            {idx + 1}
                          </td>

                          <td className="py-3 pr-4">
                            <div className="font-medium text-neutral-900 dark:text-neutral-100">
                              {driver}
                            </div>
                            <div className="text-xs text-neutral-500 dark:text-neutral-400 tabular-nums">
                              #{number}
                            </div>
                          </td>

                          <td className="py-3 pr-4">
                            <span className="inline-flex items-center rounded-full border border-neutral-200/70 bg-neutral-50 px-2 py-1 text-xs font-medium text-neutral-700 dark:border-white/10 dark:bg-white/5 dark:text-neutral-300">
                              {team}
                            </span>
                          </td>

                          <td className="py-3 pr-4 font-medium text-neutral-900 dark:text-neutral-100 tabular-nums">
                            P{grid}
                          </td>

                          <td className="py-3 pr-4 font-semibold text-neutral-900 dark:text-neutral-100 tabular-nums">
                            {score}
                          </td>

                          <td className="py-3">
                            <div className="flex items-center gap-3">
                              <div className="h-2 w-28 rounded-full bg-neutral-200 dark:bg-neutral-900 overflow-hidden">
                                <div
                                  className="h-2 rounded-full"
                                  style={{
                                    width: `${conf}%`,
                                    backgroundColor: confidenceColor(conf),
                                  }}
                                />
                              </div>
                              <div className="w-12 text-right text-xs font-semibold text-neutral-800 dark:text-neutral-200 tabular-nums">
                                {Math.round(conf)}%
                              </div>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {/* Optional analysis summary */}
              {analysis ? (
                <div className="mt-2 rounded-xl border border-neutral-200/80 bg-white/70 p-4 dark:border-white/10 dark:bg-neutral-950/50">
                  <div className="text-xs font-semibold text-neutral-700 dark:text-neutral-300">
                    Summary
                  </div>

                  {typeof analysis === "string" ? (
                    <div className="mt-2 text-sm text-neutral-600 dark:text-neutral-400 leading-relaxed">
                      {analysis}
                    </div>
                  ) : typeof analysis === "object" ? (
                    <div className="mt-3 space-y-2 text-sm">
                      {Object.entries(analysis).map(([k, v]) => (
                        <div key={k} className="flex items-start justify-between gap-4">
                          <div className="text-xs font-medium text-neutral-600 dark:text-neutral-400">
                            {k.replace(/_/g, " ")}
                          </div>
                          <div className="min-w-0 text-right text-xs text-neutral-900 dark:text-neutral-200">
                            {typeof v === "string" ||
                              typeof v === "number" ||
                              typeof v === "boolean"
                              ? String(v)
                              : Array.isArray(v)
                                ? v.join(", ")
                                : JSON.stringify(v)}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : null}
                </div>
              ) : null}
            </div>
          </Card>

          {/* Tire strategy */}
          <Card className="xl:col-span-4 p-5" clip>
            <div className="flex flex-col gap-4">
              <div>
                <h2 className="text-sm font-semibold tracking-tight">Tire strategy</h2>
                {circuitAnalysis ? (
                  <p className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">
                    Wear: {(circuitAnalysis.tire_wear_rate * 100).toFixed(0)}% • Type:{" "}
                    {circuitAnalysis.track_type}
                  </p>
                ) : (
                  <p className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">
                    Strategy suggestions for the selected circuit.
                  </p>
                )}
              </div>

              {tireStrategies.length === 0 ? (
                <div className="rounded-xl border border-neutral-200/80 bg-white/70 p-4 text-sm text-neutral-600 dark:border-white/10 dark:bg-neutral-950/50 dark:text-neutral-400">
                  No strategy available (yet).
                </div>
              ) : (
                <div className="space-y-3">
                  {tireStrategies.map((strat, idx) => (
                    <div
                      key={idx}
                      className="rounded-xl border border-neutral-200/80 bg-white/70 p-4 dark:border-white/10 dark:bg-neutral-950/50"
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div>
                          <div className="flex items-center gap-2">
                            <Badge variant={strat.rank === 1 ? "accent" : "neutral"}>
                              {strat.rank === 1 ? "Primary" : "Alternative"}
                            </Badge>
                            <div className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">
                              {String(strat.strategy_type || "")
                                .replace(/_/g, " ")
                                .toUpperCase()}
                            </div>
                          </div>

                          <div className="mt-2 text-xs text-neutral-500 dark:text-neutral-400">
                            Pit stops:{" "}
                            <span className="font-medium text-neutral-800 dark:text-neutral-200">
                              {strat.pit_stop_laps?.length
                                ? strat.pit_stop_laps.map((lap) => `Lap ${lap}`).join(", ")
                                : "None"}
                            </span>
                          </div>
                        </div>

                        <div className="text-right">
                          <div className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 tabular-nums">
                            {Math.round(strat.confidence ?? 0)}%
                          </div>
                          <div className="text-xs text-neutral-500 dark:text-neutral-400">
                            confidence
                          </div>
                        </div>
                      </div>

                      {strat.tire_sequence?.length ? (
                        <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
                          {strat.tire_sequence.map((compound, i) => (
                            <span key={i} className="inline-flex items-center gap-2">
                              <span className="rounded-full border border-neutral-200/70 bg-neutral-50 px-2 py-1 font-medium text-neutral-700 dark:border-white/10 dark:bg-white/5 dark:text-neutral-300">
                                {compound}
                              </span>
                              {i < strat.tire_sequence.length - 1 ? (
                                <span className="text-neutral-400 dark:text-neutral-600">→</span>
                              ) : null}
                            </span>
                          ))}
                        </div>
                      ) : null}

                      {strat.recommendation ? (
                        <div className="mt-3 border-t border-neutral-200/80 pt-3 text-sm text-neutral-600 dark:border-white/10 dark:text-neutral-400">
                          {strat.recommendation}
                        </div>
                      ) : null}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </Card>
        </div>
      ) : null}

      {/* Empty state */}
      {!loading && !hasPredictions && !error ? (
        <Card className="p-6" clip>
          <div className="text-center">
            <div className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">
              No data yet
            </div>
            <div className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">
              Select a race to generate pre-race predictions.
            </div>
          </div>
        </Card>
      ) : null}
    </div>
  );
}
