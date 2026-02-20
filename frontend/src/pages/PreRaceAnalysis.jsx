import "./PreRaceAnalysis.css";
import { useState, useEffect, useRef, useMemo } from "react";

import RaceSelector from "../components/RaceSelector";
import Card from "../components/ui/Card";
import Badge from "../components/ui/Badge";
import TireBadge from "../components/ui/TireBadge";

const TEAM_COLORS = {
  "Mercedes": "#00D7B6",
  "Red Bull Racing": "#4781D7",
  "Ferrari": "#ED1131",
  "McLaren": "#F47600",
  "Alpine": "#00A1E8",
  "Racing Bulls": "#6C98FF",
  "Aston Martin": "#229971",
  "Williams": "#1868DB",
  "Kick Sauber": "#01C00E",
  "Haas": "#9C9FA2",
  "Red Bull": "#4781D7",
};

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
          <h1 className="text-xl font-semibold tracking-tight">Pre-race Analysis</h1>
          <p className="text-sm text-neutral-500 dark:text-neutral-400">
            AI predictions and tire strategy per race weekend.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant="accent">Race {raceNumber}</Badge>
          {loading ? <Badge variant="warning">Loading</Badge> : null}
          {error   ? <Badge variant="danger">Error</Badge>   : null}
        </div>
      </div>

      {/* Race selector */}
      <Card className="p-4" clip>
        <RaceSelector
          selectedRace={raceNumber}
          onSelectRace={handleRaceSelect}
          variant="subtle"
        />
      </Card>

      {/* Error */}
      {error ? (
        <Card className="p-4" clip>
          <div className="flex items-start justify-between gap-3">
            <div>
              <div className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">Something went wrong</div>
              <div className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">{error}</div>
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
              <div className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">Generating analysis</div>
              <div className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">Loading models and computing predictions…</div>
            </div>
          </div>
        </Card>
      ) : null}

      {/* Main content */}
      {!loading && hasPredictions ? (
        <>
          {/* ── PODIUM PREVIEW ── */}
          <div>
            <h2 className="mb-3 text-sm font-semibold tracking-tight">Predicted Podium</h2>
            <div className="grid grid-cols-3 gap-3">
              {[
                { pred: top10[1], pos: 2, height: "pt-8" },
                { pred: top10[0], pos: 1, height: "pt-0" },
                { pred: top10[2], pos: 3, height: "pt-12" },
              ].map(({ pred, pos, height }) => {
                if (!pred) return null;
                const team = pred.team ?? "";
                const color = TEAM_COLORS[team] || "#888";
                const driver = pred.driver_name || pred.driver || "—";
                const conf = typeof pred.confidence === "number" ? Math.round(pred.confidence) : 0;
                const delta = typeof pred.grid_position === "number" ? pred.grid_position - pos : null;
                return (
                  <div key={pos} className={`flex flex-col items-center ${height}`}>
                    {/* Medal */}
                    <div
                      className="mb-2 flex h-9 w-9 items-center justify-center rounded-full text-sm font-black text-white shadow-md"
                      style={{ backgroundColor: pos === 1 ? "#F5B800" : pos === 2 ? "#9EB0C4" : "#C87533" }}
                    >
                      P{pos}
                    </div>

                    {/* Driver card */}
                    <div
                      className="relative w-full overflow-hidden rounded-xl border border-neutral-200/80 bg-white/90 p-3 text-center dark:border-white/10 dark:bg-neutral-950/60"
                      style={{ boxShadow: `0 0 0 2px ${color}30, 0 4px 16px ${color}20` }}
                    >
                      {/* team accent top bar */}
                      <div className="absolute inset-x-0 top-0 h-1 rounded-t-xl" style={{ backgroundColor: color }} />
                      <div className="mt-1 text-[11px] font-semibold uppercase tracking-widest text-neutral-500 dark:text-neutral-400">
                        {team}
                      </div>
                      <div className="mt-0.5 text-base font-bold text-neutral-900 dark:text-neutral-50 leading-tight">
                        {driver}
                      </div>
                      <div className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">
                        {conf}% confidence
                      </div>
                      {delta !== null && delta !== 0 && (
                        <div className={`mt-1 text-xs font-semibold ${delta > 0 ? "text-emerald-500" : "text-red-500"}`}>
                          {delta > 0 ? `▲ ${delta}` : `▼ ${Math.abs(delta)}`} vs grid
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* ── MOVERS STRIP ── */}
          {(() => {
            const withDelta = top10.map((p, i) => ({
              ...p,
              predictedPos: i + 1,
              delta: typeof p.grid_position === "number" ? p.grid_position - (i + 1) : 0,
            }));
            const riser = [...withDelta].sort((a, b) => b.delta - a.delta)[0];
            const faller = [...withDelta].sort((a, b) => a.delta - b.delta)[0];
            const topConf = [...withDelta].sort((a, b) => b.confidence - a.confidence)[0];
            return (
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                {[
                  { label: "Biggest Riser", driver: riser?.driver_name || riser?.driver, team: riser?.team, value: riser?.delta > 0 ? `▲ ${riser.delta} positions` : "No change", color: "text-emerald-500 dark:text-emerald-400" },
                  { label: "Biggest Faller", driver: faller?.driver_name || faller?.driver, team: faller?.team, value: faller?.delta < 0 ? `▼ ${Math.abs(faller.delta)} positions` : "No change", color: "text-red-500 dark:text-red-400" },
                  { label: "Highest Confidence", driver: topConf?.driver_name || topConf?.driver, team: topConf?.team, value: `${Math.round(topConf?.confidence ?? 0)}%`, color: "text-[rgb(var(--accent))]" },
                ].map(({ label, driver, team, value, color }) => {
                  const tc = TEAM_COLORS[team] || "#888";
                  return (
                    <Card key={label} className="p-4" clip>
                      <p className="text-xs text-neutral-500 dark:text-neutral-400 mb-1">{label}</p>
                      <div className="flex items-center gap-2">
                        <span className="inline-block h-3 w-1 rounded-full flex-shrink-0" style={{ backgroundColor: tc }} />
                        <p className="text-sm font-bold text-neutral-900 dark:text-neutral-50 truncate">{driver}</p>
                      </div>
                      <p className={`text-xs font-semibold mt-0.5 ${color}`}>{value}</p>
                    </Card>
                  );
                })}
              </div>
            );
          })()}

          {/* ── MAIN GRID ── */}
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-12">
            {/* Predictions table */}
            <Card className="xl:col-span-8 p-5" clip>
              <div className="flex flex-col gap-4">
                <div>
                  <h2 className="text-sm font-semibold tracking-tight">AI Predictions</h2>
                  <p className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">Top 10 expected finishing order</p>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-neutral-200/80 dark:border-white/10">
                        <th className="py-3 pr-3 text-left text-xs font-semibold text-neutral-500 dark:text-neutral-400">#</th>
                        <th className="py-3 pr-4 text-left text-xs font-semibold text-neutral-500 dark:text-neutral-400">Driver</th>
                        <th className="py-3 pr-4 text-left text-xs font-semibold text-neutral-500 dark:text-neutral-400">Grid</th>
                        <th className="py-3 pr-4 text-left text-xs font-semibold text-neutral-500 dark:text-neutral-400">Δ</th>
                        <th className="py-3 text-left text-xs font-semibold text-neutral-500 dark:text-neutral-400">Confidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {top10.map((pred, idx) => {
                        const driver  = pred.driver_name || pred.driver || "Unknown";
                        const team    = pred.team ?? "—";
                        const grid    = pred.grid_position ?? null;
                        const conf    = typeof pred.confidence === "number" ? Math.max(0, Math.min(100, pred.confidence)) : 0;
                        const delta   = grid !== null ? grid - (idx + 1) : null;
                        const color   = TEAM_COLORS[team] || "#888";

                        const confValues2 = top10.map(p => Number(p.confidence)).filter(n => Number.isFinite(n));
                        const cMin = confValues2.length ? Math.min(...confValues2) : 0;
                        const cMax = confValues2.length ? Math.max(...confValues2) : 100;
                        const t = cMax === cMin ? 0.5 : (conf - cMin) / (cMax - cMin);
                        const barColor = `hsl(${Math.round(15 + t * 85)} 85% 52%)`;

                        return (
                          <tr key={idx} className="relative border-b border-neutral-100 hover:bg-neutral-50/70 dark:border-white/5 dark:hover:bg-white/[0.04]">
                            <td className="py-3 pr-3">
                              <div className="flex items-center gap-2">
                                <span className="inline-block h-4 w-1 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
                                <span className="font-bold text-neutral-900 dark:text-neutral-100 tabular-nums">{idx + 1}</span>
                              </div>
                            </td>

                            <td className="py-3 pr-4">
                              <div className="font-medium text-neutral-900 dark:text-neutral-100 leading-tight">{driver}</div>
                              <div className="text-[11px] text-neutral-500 dark:text-neutral-400">{team}</div>
                            </td>

                            <td className="py-3 pr-4 tabular-nums text-neutral-700 dark:text-neutral-300 font-medium">
                              {grid !== null ? `P${grid}` : "—"}
                            </td>

                            <td className="py-3 pr-4 tabular-nums font-semibold">
                              {delta === null ? (
                                <span className="text-neutral-400">—</span>
                              ) : delta > 0 ? (
                                <span className="text-emerald-500">▲{delta}</span>
                              ) : delta < 0 ? (
                                <span className="text-red-500">▼{Math.abs(delta)}</span>
                              ) : (
                                <span className="text-neutral-400">—</span>
                              )}
                            </td>

                            <td className="py-3">
                              <div className="flex items-center gap-2.5">
                                <div className="h-2 w-24 rounded-full bg-neutral-200 dark:bg-neutral-900 overflow-hidden">
                                  <div className="h-2 rounded-full transition-all duration-500" style={{ width: `${conf}%`, backgroundColor: barColor }} />
                                </div>
                                <span className="w-10 text-right text-xs font-semibold text-neutral-800 dark:text-neutral-200 tabular-nums">
                                  {Math.round(conf)}%
                                </span>
                              </div>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                {/* Analysis summary */}
                {analysis ? (
                  <div className="rounded-xl border border-neutral-200/80 bg-white/70 p-4 dark:border-white/10 dark:bg-neutral-950/50">
                    <div className="text-xs font-semibold text-neutral-700 dark:text-neutral-300 mb-2">Summary</div>
                    {typeof analysis === "string" ? (
                      <p className="text-sm text-neutral-600 dark:text-neutral-400 leading-relaxed">{analysis}</p>
                    ) : typeof analysis === "object" ? (
                      <div className="space-y-1.5 text-xs">
                        {Object.entries(analysis).map(([k, v]) => (
                          <div key={k} className="flex items-start justify-between gap-4">
                            <span className="font-medium text-neutral-500 dark:text-neutral-400">{k.replace(/_/g, " ")}</span>
                            <span className="text-right text-neutral-900 dark:text-neutral-200">
                              {typeof v === "string" || typeof v === "number" || typeof v === "boolean"
                                ? String(v)
                                : Array.isArray(v) ? v.join(", ") : JSON.stringify(v)}
                            </span>
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
                  <h2 className="text-sm font-semibold tracking-tight">Tire Strategy</h2>
                  {circuitAnalysis ? (
                    <p className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">
                      Wear: {(circuitAnalysis.tire_wear_rate * 100).toFixed(0)}% · {circuitAnalysis.track_type}
                    </p>
                  ) : (
                    <p className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">
                      Strategy suggestions for the selected circuit.
                    </p>
                  )}
                </div>

                {tireStrategies.length === 0 ? (
                  <div className="rounded-xl border border-neutral-200/80 bg-white/70 p-4 text-sm text-neutral-600 dark:border-white/10 dark:bg-neutral-950/50 dark:text-neutral-400">
                    No strategy available yet.
                  </div>
                ) : (
                  <div className="space-y-3">
                    {tireStrategies.map((strat, idx) => (
                      <div key={idx} className="rounded-xl border border-neutral-200/80 bg-white/70 p-4 dark:border-white/10 dark:bg-neutral-950/50">
                        <div className="flex items-start justify-between gap-3">
                          <div>
                            <div className="flex items-center gap-2">
                              <Badge variant={strat.rank === 1 ? "accent" : "neutral"}>
                                {strat.rank === 1 ? "Primary" : "Alternative"}
                              </Badge>
                              <span className="text-xs font-semibold text-neutral-700 dark:text-neutral-300">
                                {String(strat.strategy_type || "").replace(/_/g, " ").toUpperCase()}
                              </span>
                            </div>
                            <div className="mt-1.5 text-xs text-neutral-500 dark:text-neutral-400">
                              Pit stops:{" "}
                              <span className="font-medium text-neutral-800 dark:text-neutral-200">
                                {strat.pit_stop_laps?.length
                                  ? strat.pit_stop_laps.map((lap) => `Lap ${lap}`).join(", ")
                                  : "None"}
                              </span>
                            </div>
                          </div>
                          <div className="text-right flex-shrink-0">
                            <div className="text-lg font-bold text-neutral-900 dark:text-neutral-100 tabular-nums">
                              {Math.round(strat.confidence ?? 0)}%
                            </div>
                            <div className="text-[11px] text-neutral-500 dark:text-neutral-400">confidence</div>
                          </div>
                        </div>

                        {strat.tire_sequence?.length ? (
                          <div className="mt-3 flex flex-wrap items-center gap-1.5">
                            {strat.tire_sequence.map((compound, i) => (
                              <div key={i} className="flex items-center gap-1.5">
                                <TireBadge compound={compound} />
                                {i < strat.tire_sequence.length - 1 && (
                                  <span className="text-neutral-400 text-xs">→</span>
                                )}
                              </div>
                            ))}
                          </div>
                        ) : null}

                        {strat.recommendation ? (
                          <div className="mt-3 border-t border-neutral-200/80 pt-3 text-xs leading-relaxed text-neutral-600 dark:border-white/10 dark:text-neutral-400">
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
        </>
      ) : null}

      {/* Empty state */}
      {!loading && !hasPredictions && !error ? (
        <Card className="p-6" clip>
          <div className="text-center">
            <div className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">No data yet</div>
            <div className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">
              Select a race to generate pre-race predictions.
            </div>
          </div>
        </Card>
      ) : null}
    </div>
  );
}

