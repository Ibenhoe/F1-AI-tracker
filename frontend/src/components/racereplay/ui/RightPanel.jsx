// src/components/racereplay/ui/RightPanel.jsx
import React, { useMemo } from "react";
import Card from "../../ui/Card";

import ReplayLeaderboard from "../../ReplayLeaderboard";
import DriverInfoPanel from "../../DriverInfoPanel";

import SegmentedControl from "./controls/SegmentedControl";

function EmptyState({ title, subtitle }) {
  return (
    <div className="flex h-full items-center justify-center">
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

function MiniTag({ children }) {
  return (
    <span
      className={[
        "inline-flex items-center rounded-full px-2 py-0.5",
        "text-[10px] font-semibold uppercase tracking-widest",
        "bg-black/[0.03] dark:bg-white/[0.06]",
        "ring-1 ring-black/5 dark:ring-white/10",
        "text-neutral-600 dark:text-neutral-300",
      ].join(" ")}
    >
      {children}
    </span>
  );
}

function EventRow({ event }) {
  const isPit = event.type === "pit_stop";
  const isRet = event.type === "retirement";

  const railColor = isPit
    ? "rgba(var(--accent), 0.70)"
    : isRet
      ? "rgb(239 68 68 / 0.80)"
      : "rgba(var(--accent), 0.35)";

  const tag = isPit ? "PIT" : isRet ? "RET" : "FLAG";

  return (
    <div
      className={[
        "relative overflow-hidden rounded-2xl",
        "bg-white dark:bg-neutral-950/40",
        "ring-1 ring-black/5 dark:ring-white/10",
        "px-4 py-3",
      ].join(" ")}
    >
      {/* accent rail */}
      <div
        className="absolute left-0 top-0 h-full w-1 opacity-90"
        style={{ background: railColor }}
        aria-hidden="true"
      />

      <div className="pl-2">
        <div className="flex items-start justify-between gap-3">
          <p className="min-w-0 text-sm font-semibold text-neutral-900 dark:text-neutral-50">
            {event.message}
          </p>
          <MiniTag>{tag}</MiniTag>
        </div>

        <p className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">
          Lap <span className="tabular-nums">{event.lap}</span>
        </p>
      </div>
    </div>
  );
}

export default function RightPanel({
  rightPanelTab,
  setRightPanelTab,
  currentFrame,
  selectedDriver,
  setSelectedDriver,
  currentLap,
  totalLaps,
  showTelemetry,
  raceEvents,
}) {
  const tabs = useMemo(
    () => [
      { id: "standings", label: "Standings" },
      { id: "driver", label: "Driver" },
      { id: "events", label: "Events" },
    ],
    []
  );

  return (
    <Card className="p-5" clip>
      <div className="flex h-full min-w-0 flex-col gap-4 overflow-hidden">
        <div className="flex items-center justify-between gap-3">
          <div className="min-w-0">
            <div className="text-[11px] font-semibold uppercase tracking-widest text-neutral-500 dark:text-neutral-400">
              Live
            </div>
            <div className="mt-1 text-sm font-semibold text-neutral-900 dark:text-neutral-50">
              Race data
            </div>
          </div>

          <div className="shrink-0">
            <span
              className={[
                "inline-flex items-center rounded-full px-3 py-1",
                "text-xs font-semibold tabular-nums",
                "bg-black/[0.03] dark:bg-white/[0.06]",
                "ring-1 ring-black/5 dark:ring-white/10",
                "text-neutral-700 dark:text-neutral-200",
              ].join(" ")}
            >
              Lap {currentLap}
              {totalLaps ? `/${totalLaps}` : ""}
            </span>
          </div>
        </div>

        <SegmentedControl
          value={rightPanelTab}
          onChange={setRightPanelTab}
          ariaLabel="Replay right panel tabs"
          items={tabs}
        />

        {/* Content: scroll ONLY inside this area */}
        <div className="min-h-0 min-w-0 flex-1 overflow-hidden rounded-2xl ring-1 ring-black/5 dark:ring-white/10">
          {rightPanelTab === "standings" ? (
            <div className="h-full min-h-0 overflow-y-auto">
              <ReplayLeaderboard
                drivers={currentFrame?.drivers || []}
                selectedDriver={selectedDriver}
                onDriverSelect={setSelectedDriver}
                currentLap={currentLap}
                totalLaps={totalLaps}
              />
            </div>
          ) : null}

          {rightPanelTab === "driver" ? (
            selectedDriver && currentFrame ? (
              <div className="h-full min-h-0 overflow-y-auto">
                <DriverInfoPanel
                  driver={selectedDriver}
                  driverData={currentFrame.drivers[selectedDriver]}
                  frame={currentFrame}
                  showTelemetry={showTelemetry}
                />
              </div>
            ) : (
              <EmptyState
                title="No driver selected"
                subtitle="Click a driver on the track or standings."
              />
            )
          ) : null}

          {rightPanelTab === "events" ? (
            <div className="h-full min-h-0 overflow-y-auto">
              {Array.isArray(raceEvents) && raceEvents.length > 0 ? (
                <div className="space-y-3">
                  {raceEvents.slice(0, 25).map((event) => (
                    <EventRow key={event.id} event={event} />
                  ))}

                  {raceEvents.length > 25 ? (
                    <p className="pt-1 text-xs text-neutral-500 dark:text-neutral-400">
                      Showing latest <span className="tabular-nums">25</span>{" "}
                      events.
                    </p>
                  ) : null}
                </div>
              ) : (
                <EmptyState
                  title="No events yet"
                  subtitle="Events will appear as the race unfolds."
                />
              )}
            </div>
          ) : null}
        </div>
      </div>
    </Card>
  );
}