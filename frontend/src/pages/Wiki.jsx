// src/pages/Wiki.jsx
import { useCallback, useEffect, useMemo, useState } from "react";

import {
  CIRCUIT_INFO,
  DRIVER_MAPPING,
  DRIVER_WIKI,
  DriverModal,
  WikiTable,
  CircuitCard,
  useWikipediaSummary,
  useWikiRaces,
  useWikiSessionData,
} from "../components/wiki";

import RaceSelector from "../components/dashboard/RaceSelector";
import Card from "../components/ui/Card";
import Badge from "../components/ui/Badge";

const FIRST_F1_YEAR = 1950;

function buildYears() {
  const currentYear = new Date().getFullYear();
  return Array.from(
    { length: currentYear - FIRST_F1_YEAR + 1 },
    (_, i) => currentYear - i
  );
}

const YEARS = buildYears();

/* ---------- SegmentedControl (Dashboard style) ---------- */
function SegmentedControl({ value, onChange, items, ariaLabel }) {
  const activeIndex = Math.max(0, items.findIndex((i) => i.id === value));

  return (
    <div
      className={[
        "relative inline-flex w-full items-stretch justify-center",
        "rounded-2xl p-1",
        // remove the gray track
        "bg-transparent",
        // keep only a very subtle container outline
        "ring-1 ring-black/5 dark:ring-white/10",
      ].join(" ")}
      role="tablist"
      aria-label={ariaLabel}
    >
      {/* Active pill (accent) */}
      <div
        className={[
          "pointer-events-none absolute top-1 bottom-1 left-1",
          "rounded-2xl",
          "bg-[rgb(var(--accent))]",
          "ring-1 ring-black/10 dark:ring-white/10",
          "transition-transform duration-200 ease-out",
        ].join(" ")}
        style={{
          width: `calc((100% - 0.5rem) / ${items.length})`,
          transform: `translateX(calc(${activeIndex} * 100%))`,
        }}
        aria-hidden="true"
      />

      {items.map((item) => {
        const active = value === item.id;

        return (
          <button
            key={item.id}
            type="button"
            onClick={() => onChange(item.id)}
            role="tab"
            aria-selected={active}
            className={[
              "relative z-10 flex-1 min-w-0",
              "inline-flex items-center justify-center gap-2",
              "rounded-2xl px-3 py-1.5 text-sm font-semibold",
              "transition-colors",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[rgb(var(--accent))] focus-visible:ring-offset-2 focus-visible:ring-offset-transparent",
              active
                ? "text-[rgb(var(--accent-fg))]"
                : "text-neutral-600 dark:text-neutral-300 hover:text-neutral-900 dark:hover:text-neutral-50 hover:bg-black/[0.03] dark:hover:bg-white/[0.05]",
            ].join(" ")}
          >
            <span className="whitespace-nowrap">{item.label}</span>
            {item.trailing ? item.trailing : null}
          </button>
        );
      })}
    </div>
  );
}

function EmptyState({ title, subtitle }) {
  return (
    <div className="flex min-h-[220px] items-center justify-center">
      <div className="text-center">
        <p className="text-sm font-medium text-neutral-900 dark:text-neutral-50">
          {title}
        </p>
        {subtitle ? (
          <p className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">
            {subtitle}
          </p>
        ) : null}
      </div>
    </div>
  );
}

export default function Wiki() {
  const [selectedYear, setSelectedYear] = useState(2024);
  const [selectedRaceId, setSelectedRaceId] = useState("");
  const [viewType, setViewType] = useState("race");

  const { races, error: racesError, defaultRaceId } = useWikiRaces(selectedYear);

  const { tableData, loading, error: sessionError } = useWikiSessionData(
    selectedRaceId,
    viewType
  );

  const error = racesError || sessionError;

  useEffect(() => {
    setSelectedRaceId(defaultRaceId ? String(defaultRaceId) : "");
  }, [defaultRaceId]);

  const formatName = useCallback((name) => {
    if (!name) return "";
    const lowerName = String(name).toLowerCase();
    if (DRIVER_MAPPING[lowerName]) return DRIVER_MAPPING[lowerName];

    return String(name)
      .replace(/_/g, " ")
      .split(" ")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(" ");
  }, []);

  const circuitInfo = useMemo(() => {
    if (!selectedRaceId || races.length === 0) return null;

    const race = races.find((r) => String(r.raceId) === String(selectedRaceId));
    if (!race) return null;

    const name = race.name || "";

    return (
      CIRCUIT_INFO[name] ||
      Object.entries(CIRCUIT_INFO).find(([k]) => {
        const a = name.toLowerCase();
        const b = k.toLowerCase();
        return a.includes(b) || b.includes(a);
      })?.[1] ||
      null
    );
  }, [selectedRaceId, races]);

  const { data: circuitWiki } = useWikipediaSummary(circuitInfo?.wikiPage);
  const circuitImage = circuitWiki?.thumbnail?.source || null;

  const [selectedDriver, setSelectedDriver] = useState(null);

  const driverWikiTitle = useMemo(() => {
    if (!selectedDriver) return null;
    return DRIVER_WIKI[selectedDriver.driverRef] || selectedDriver.fullName;
  }, [selectedDriver]);

  const { data: driverWiki, loading: driverWikiLoading } =
    useWikipediaSummary(driverWikiTitle);

  const handleDriverClick = useCallback(
    (row) => {
      setSelectedDriver({
        driverRef: row.driver?.toString().toLowerCase(),
        fullName: formatName(row.driver),
        team: row.team,
        row,
      });
    },
    [formatName]
  );

  // RaceSelector items voor Wiki (volgorde = API)
  const raceItems = useMemo(() => {
    if (!Array.isArray(races)) return [];
    return races.map((r) => ({
      id: r.raceId,
      name: r.name ? String(r.name) : `Race ${r.raceId}`,
      meta: r.round != null ? `Round ${r.round}` : "",
    }));
  }, [races]);

  // Circuit tabs rechts
  const [circuitTab, setCircuitTab] = useState("overview"); // overview | logo

  useEffect(() => {
    // als je van race verandert: zet terug op overview (rustiger UX)
    setCircuitTab("overview");
  }, [selectedRaceId]);

  const Tile = ({ title, value, accentColor = "rgb(var(--accent))" }) => (
    <Card className="relative overflow-hidden p-5" clip>
      {/* Subtle top accent line (same as Dashboard tiles) */}
      <div
        className="absolute left-0 top-0 h-1 w-full opacity-80"
        style={{ background: accentColor }}
        aria-hidden="true"
      />

      <p className="text-[11px] font-semibold uppercase tracking-widest text-neutral-500 dark:text-neutral-500">
        {title}
      </p>

      <div className="mt-3 text-lg font-semibold text-neutral-900 dark:text-neutral-50">
        {value ?? "—"}
      </div>
    </Card>
  );

  const ready = !loading && !error;

  return (
    <>
      <div className="space-y-6">
        {/* HEADER */}
        <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div className="space-y-1">
            <h1 className="text-2xl font-semibold tracking-tight text-neutral-900 dark:text-neutral-50">
              Wiki
            </h1>
            <p className="text-sm text-neutral-500 dark:text-neutral-400">
              Historical race results, qualifying sessions and starting grids.
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
              <Badge variant="accent">{selectedYear}</Badge>
              <Badge variant={loading ? "warning" : "neutral"}>
                {loading ? "Loading" : "Ready"}
              </Badge>
              {error ? <Badge variant="danger">Error</Badge> : null}
            </div>
          </div>
        </div>

        {/* TOP GRID (Dashboard structuur) */}
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
          {/* LEFT: RaceSelector + year selector */}
          <Card className="lg:col-span-4 p-5" clip>
            <div className="h-[320px] min-h-0">
              <RaceSelector
                mode="prerace"
                visibleRows={5}
                selectedRace={selectedRaceId}
                onSelectRace={(id) => setSelectedRaceId(String(id))}
                disabled={false}
                raceLoading={loading && raceItems.length === 0}
                raceReady={raceItems.length > 0}
                raceRunning={false}
                raceEverStarted={false}
                speed={1}
                items={raceItems}
                years={YEARS}
                selectedYear={selectedYear}
                onYearChange={setSelectedYear}
              />
            </div>
          </Card>

          {/* RIGHT: Circuit info tabs */}
          <Card className="lg:col-span-8 p-5" clip>
            <div className="flex h-[320px] min-h-0 flex-col gap-4 overflow-hidden">
              {/* Tabs (full width like Dashboard) */}
              <SegmentedControl
                value={circuitTab}
                onChange={setCircuitTab}
                ariaLabel="Circuit tabs"
                items={[
                  { id: "overview", label: "Overview" },
                  { id: "logo", label: "Logo" },
                ]}
              />

              {/* Content area (no scroll) */}
              <div className="min-h-0 flex-1 overflow-hidden">
                {!ready ? (
                  <div className="h-full">
                    <EmptyState
                      title={error ? "Error" : "Loading"}
                      subtitle={
                        error
                          ? "Check your backend and try again."
                          : "Circuit details are loading…"
                      }
                    />
                  </div>
                ) : !circuitInfo ? (
                  <div className="h-full">
                    <EmptyState title="No circuit info" subtitle="Select a race first." />
                  </div>
                ) : (
                  <CircuitCard
                    tab={circuitTab}              // "overview" | "logo"
                    circuitInfo={circuitInfo}
                    circuitImage={circuitImage}
                  />
                )}
              </div>
            </div>
          </Card>
        </div>

        {/* STATS TILES */}
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Tile title="Track length" value={circuitInfo?.length} accentColor="rgb(var(--accent))" />
          <Tile title="Race laps" value={circuitInfo?.laps} accentColor="rgb(var(--accent))" />
          <Tile title="Distance" value={circuitInfo?.distance} accentColor="rgb(var(--accent))" />
          <Tile title="First GP" value={circuitInfo?.firstGP} accentColor="rgb(var(--accent))" />
        </div>

        {/* MAIN PANEL: Race/Grid/Qualifying + tabel */}
        <Card className="p-5" clip>
          <div className="mb-4 flex flex-col gap-3 sm:grid sm:grid-cols-3 sm:items-end">
            <div className="space-y-1">
              <p className="text-[11px] font-semibold uppercase tracking-widest text-neutral-400 dark:text-neutral-600">
                Results
              </p>
              <h2 className="text-base font-semibold tracking-tight text-neutral-900 dark:text-neutral-50">
                {viewType === "race" ? "Race Classification" : "Qualifying Results"}
              </h2>
              <p className="text-xs text-neutral-500 dark:text-neutral-400">
                Click on a driver for additional information ⓘ
              </p>
            </div>

            <div className="sm:flex sm:justify-center">
              <SegmentedControl
                value={viewType}
                onChange={setViewType}
                ariaLabel="Wiki session tabs"
                items={[
                  { id: "race", label: "Race Results" },
                  { id: "qualifying", label: "Qualifying Results" },
                ]}
              />
            </div>
          </div>

          <div className="min-w-0">
            <WikiTable
              viewType={viewType}
              loading={loading}
              tableData={tableData}
              formatName={formatName}
              onDriverClick={handleDriverClick}
            />
          </div>
        </Card>
      </div>

      {/* Driver modal (later final polish) */}
      <DriverModal
        selectedDriver={selectedDriver}
        onClose={() => setSelectedDriver(null)}
        driverWiki={driverWiki}
        driverWikiLoading={driverWikiLoading}
      />
    </>
  );
}