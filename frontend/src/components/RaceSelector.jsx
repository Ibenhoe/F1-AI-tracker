import { useEffect, useMemo, useRef, useState } from "react";
import { Loader2, Check, Play, Pause, ChevronLeft, ChevronRight } from "lucide-react";

const RACES = {
  1: "Bahrain",
  2: "Saudi Arabia",
  3: "Australia",
  4: "Japan",
  5: "China",
  6: "Miami",
  7: "Emilia Romagna",
  8: "Monaco",
  9: "Canada",
  10: "Spain",
  11: "Austria",
  12: "United Kingdom",
  13: "Hungary",
  14: "Belgium",
  15: "Netherlands",
  16: "Italy",
  17: "Azerbaijan",
  18: "Singapore",
  19: "Austin",
  20: "Mexico",
  21: "Brazil",
  22: "Las Vegas",
  23: "Qatar",
  24: "Abu Dhabi",
};

const ITEM_H = 44;
const LOOPS = 3;

function Pill({ children }) {
  return (
    <span className="inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs ring-1 ring-black/5 bg-black/[0.02] text-neutral-700 dark:ring-white/10 dark:bg-white/[0.04] dark:text-neutral-200">
      {children}
    </span>
  );
}

function IconButton({
  onClick,
  disabled,
  title,
  active = false,
  children,
  accentIcon = false,
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={title}
      title={title}
      className={[
        "inline-flex h-10 w-10 items-center justify-center rounded-2xl",
        "transition-colors duration-150",
        "active:scale-[0.98]",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[rgb(var(--accent))] focus-visible:ring-offset-2",
        "focus-visible:ring-offset-white dark:focus-visible:ring-offset-neutral-950",
        disabled ? "opacity-40 cursor-not-allowed" : "cursor-pointer",
        active
          ? "bg-[rgb(var(--accent))] text-[rgb(var(--accent-fg))]"
          : "bg-transparent hover:bg-black/[0.03] dark:hover:bg-white/[0.05]",
        "ring-1 ring-black/5 dark:ring-white/10",
        !active && accentIcon ? "text-[rgb(var(--accent))]" : "",
      ].join(" ")}
    >
      {children}
    </button>
  );
}

function SpeedPill({ value, onChange, disabled }) {
  const options = [1, 2, 4, 8];

  return (
    <div
      className={[
        "inline-flex items-center gap-1 rounded-2xl p-1",
        "bg-transparent",
        "ring-1 ring-black/5 dark:ring-white/10",
        disabled ? "opacity-60" : "",
      ].join(" ")}
      aria-label="Simulation speed"
      role="group"
    >
      {options.map((s) => {
        const active = Number(value) === s;

        return (
          <button
            key={s}
            type="button"
            disabled={disabled}
            onClick={() => onChange?.(s)}
            className={[
              "rounded-xl px-2.5 py-1 text-xs font-semibold tabular-nums transition",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[rgb(var(--accent))]",
              active
                ? "bg-[rgb(var(--accent))] text-[rgb(var(--accent-fg))]"
                : "text-neutral-600 hover:bg-black/[0.03] dark:text-neutral-300 dark:hover:bg-white/[0.05]",
            ].join(" ")}
            aria-pressed={active}
            title={`x${s}`}
          >
            x{s}
          </button>
        );
      })}
    </div>
  );
}

function YearStepper({ value, options, onChange, disabled }) {
  const years = Array.isArray(options) ? [...options] : [];
  years.sort((a, b) => Number(b) - Number(a)); // DESC: 2026 ... 1950

  const current = Number(value);
  const idx = years.findIndex((y) => Number(y) === current);

  // In DESC order:
  // left = newer (idx-1), right = older (idx+1)
  const newer = idx > 0 ? years[idx - 1] : null;
  const older = idx >= 0 && idx < years.length - 1 ? years[idx + 1] : null;

  const pillBase = [
    "h-10 rounded-2xl px-4",
    "inline-flex items-center justify-center",
    "text-sm font-semibold tabular-nums",
    "ring-1 ring-black/5 dark:ring-white/10",
  ].join(" ");

  return (
    <div
      className={[
        "w-full",
        "flex items-center gap-2",
        "rounded-2xl p-1",
        "bg-transparent",
        "ring-1 ring-black/5 dark:ring-white/10",
        disabled ? "opacity-60" : "",
      ].join(" ")}
      aria-label="Season year"
      role="group"
    >
      {/* Left arrow = newer year */}
      <button
        type="button"
        disabled={disabled || newer == null}
        onClick={() => newer != null && onChange?.(Number(newer))}
        className={[
          "inline-flex h-10 w-10 items-center justify-center rounded-2xl",
          "transition-colors duration-150",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[rgb(var(--accent))]",
          disabled || newer == null
            ? "opacity-40 cursor-not-allowed"
            : "hover:bg-black/[0.03] dark:hover:bg-white/[0.05]",
        ].join(" ")}
        title="Newer year"
        aria-label="Newer year"
      >
        <ChevronLeft size={18} />
      </button>

      {/* Prev (newer) year label */}
      <div
        className={[
          pillBase,
          "flex-1",
          "bg-transparent",
          "text-neutral-500 dark:text-neutral-400",
        ].join(" ")}
        title={newer == null ? "" : String(newer)}
        aria-hidden={newer == null ? "true" : "false"}
      >
        {newer ?? "—"}
      </div>

      {/* Current year (accent) */}
      <div
        className={[
          pillBase,
          "min-w-[92px]",
          "bg-[rgb(var(--accent))] text-[rgb(var(--accent-fg))]",
        ].join(" ")}
        aria-live="polite"
      >
        {current}
      </div>

      {/* Next (older) year label */}
      <div
        className={[
          pillBase,
          "flex-1",
          "bg-transparent",
          "text-neutral-500 dark:text-neutral-400",
        ].join(" ")}
        title={older == null ? "" : String(older)}
        aria-hidden={older == null ? "true" : "false"}
      >
        {older ?? "—"}
      </div>

      {/* Right arrow = older year */}
      <button
        type="button"
        disabled={disabled || older == null}
        onClick={() => older != null && onChange?.(Number(older))}
        className={[
          "inline-flex h-10 w-10 items-center justify-center rounded-2xl",
          "transition-colors duration-150",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[rgb(var(--accent))]",
          disabled || older == null
            ? "opacity-40 cursor-not-allowed"
            : "hover:bg-black/[0.03] dark:hover:bg-white/[0.05]",
        ].join(" ")}
        title="Older year"
        aria-label="Older year"
      >
        <ChevronRight size={18} />
      </button>
    </div>
  );
}

export default function RaceSelector({
  mode = "dashboard", // "dashboard" | "prerace"
  selectedRace,
  onRaceChange,
  onSelectRace,
  disabled,

  raceLoading = false,
  raceReady = false,

  raceRunning = false,
  raceEverStarted = false,
  speed = 1,
  onStart,
  onPause,
  onResume,
  onSpeedChange,

  visibleRows,

  items,

  years,
  selectedYear,
  onYearChange,
  yearDisabled,
}) {
  const showTransport = mode === "dashboard";

  const showYearSelector =
    Array.isArray(years) &&
    years.length > 0 &&
    typeof selectedYear === "number" &&
    typeof onYearChange === "function";

  const VISIBLE =
    typeof visibleRows === "number" && Number.isFinite(visibleRows)
      ? Math.max(3, Math.min(9, Math.round(visibleRows)))
      : mode === "prerace"
        ? 6
        : 5;

  const SELECT_SLOT = Math.floor((VISIBLE - 1) / 2);

  const baseList = useMemo(() => {
    if (Array.isArray(items) && items.length > 0) {
      return items.map((it, idx) => ({
        id: it.id,
        name: it.name ?? it.label ?? `Item ${idx + 1}`,
        meta: it.meta ?? "",
      }));
    }
    return Object.entries(RACES).map(([id, name]) => ({ id: Number(id), name }));
  }, [items]);

  const N = baseList.length;

  const normalizeIndex = (i) => {
    if (N <= 0) return 0;
    return ((i % N) + N) % N;
  };

  const loopList = useMemo(() => {
    const out = [];
    for (let i = 0; i < LOOPS; i++) out.push(...baseList);
    return out;
  }, [baseList]);

  const selectedId = selectedRace == null ? "" : String(selectedRace);

  // Debounce to avoid rapid API calls while dragging
  const debounceTimerRef = useRef(null);
  const pendingRaceRef = useRef(null);

  const emit = (id) => {
    if (disabled) return;

    if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current);
    pendingRaceRef.current = id;

    debounceTimerRef.current = setTimeout(() => {
      const finalRace = pendingRaceRef.current;
      if (finalRace !== null) {
        if (onSelectRace) onSelectRace(finalRace);
        else if (onRaceChange) onRaceChange(finalRace);
      }
      debounceTimerRef.current = null;
      pendingRaceRef.current = null;
    }, 350);
  };

  const padTop = SELECT_SLOT * ITEM_H;

  // infinite normalization (no jumps)
  const oneLoopPx = N * ITEM_H;
  const centerLoopStartPx = oneLoopPx;

  const normalizeOffset = (offPx) => {
    if (oneLoopPx <= 0) return 0;
    let x = offPx - centerLoopStartPx;
    x = ((x % oneLoopPx) + oneLoopPx) % oneLoopPx;
    return centerLoopStartPx + x;
  };

  const clampIndex = (idx) => Math.max(0, Math.min(loopList.length - 1, idx));
  const idxFromOffset = (offPx) => clampIndex(Math.round(offPx / ITEM_H));

  const getTargetOffsetForSelected = () => {
    const baseIdx = baseList.findIndex((r) => String(r.id) === String(selectedRace));
    if (baseIdx < 0) return normalizeOffset(centerLoopStartPx);
    return normalizeOffset(centerLoopStartPx + baseIdx * ITEM_H);
  };

  // offset in px (initialize at correct position to avoid first render jump)
  const [offsetPx, _setOffsetPx] = useState(() => getTargetOffsetForSelected());
  const offsetRef = useRef(offsetPx);

  // suppress transition for 1 frame when syncing from parent
  const suppressAnimRef = useRef(true);

  const setOffsetPx = (next) => {
    offsetRef.current = next;
    _setOffsetPx(next);
  };

  const snapToNearest = (offPx) => {
    const idx = idxFromOffset(offPx);
    const snapped = idx * ITEM_H;
    setOffsetPx(snapped);

    const baseIdx = normalizeIndex(idx);
    const row = baseList[baseIdx];
    if (!row) return;

    const id = row.id;
    if (String(id) !== String(selectedRace)) emit(id);
  };

  // sync from parent to middle loop (NO animation)
  useEffect(() => {
    if (N <= 0) return;

    const baseIdx = baseList.findIndex((r) => String(r.id) === String(selectedRace));
    if (baseIdx < 0) return;

    const target = normalizeOffset(centerLoopStartPx + baseIdx * ITEM_H);

    // suppress animation for one paint so it doesn't "roll"
    suppressAnimRef.current = true;
    setOffsetPx(target);
    requestAnimationFrame(() => {
      suppressAnimRef.current = false;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedId, baseList]);

  useEffect(() => {
    // after initial paint, allow animations normally
    const t = requestAnimationFrame(() => {
      suppressAnimRef.current = false;
    });
    return () => cancelAnimationFrame(t);
  }, []);

  useEffect(() => {
    return () => {
      if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current);
    };
  }, []);

  // drag state
  const isDraggingRef = useRef(false);
  const movedRef = useRef(false);
  const startYRef = useRef(0);
  const startOffsetRef = useRef(0);
  const pointerIdRef = useRef(null);
  const surfaceRef = useRef(null);

  const onPointerDown = (e) => {
    if (disabled) return;

    isDraggingRef.current = true;
    movedRef.current = false;
    pointerIdRef.current = e.pointerId;

    startYRef.current = e.clientY;
    startOffsetRef.current = offsetRef.current;

    e.currentTarget.setPointerCapture?.(e.pointerId);
  };

  const onPointerMove = (e) => {
    if (!isDraggingRef.current) return;

    const dy = e.clientY - startYRef.current;
    if (Math.abs(dy) > 4) movedRef.current = true;

    const raw = startOffsetRef.current - dy;
    const next = normalizeOffset(raw);
    setOffsetPx(next);
  };

  const endDrag = () => {
    if (!isDraggingRef.current) return;
    isDraggingRef.current = false;

    const wrapped = normalizeOffset(offsetRef.current);
    setOffsetPx(wrapped);
    snapToNearest(wrapped);
  };

  const onPointerUp = (e) => {
    if (pointerIdRef.current !== null) {
      try {
        e.currentTarget.releasePointerCapture?.(pointerIdRef.current);
      } catch { }
    }
    pointerIdRef.current = null;

    // tap-to-select within wheel (no drag)
    if (!disabled && isDraggingRef.current && movedRef.current === false) {
      const surface = surfaceRef.current;
      if (surface) {
        const rect = surface.getBoundingClientRect();
        const y = e.clientY - rect.top;
        const slot = Math.max(0, Math.min(VISIBLE - 1, Math.floor(y / ITEM_H)));
        const deltaSlots = slot - SELECT_SLOT;

        const next = normalizeOffset(offsetRef.current + deltaSlots * ITEM_H);
        setOffsetPx(next);
        snapToNearest(next);
      }
    }

    endDrag();
  };

  const onLostPointerCapture = () => {
    pointerIdRef.current = null;
    endDrag();
  };

  const canTransport = !disabled && raceReady && !raceLoading;
  const canStart = canTransport && !raceRunning && !raceEverStarted;
  const canPause = canTransport && raceRunning;
  const canResume = canTransport && !raceRunning && raceEverStarted;

  // Primary action highlight (action-based, not state-based)
  const playPrimary = canTransport && !raceRunning;
  const pausePrimary = canTransport && raceRunning;

  return (
    <div
      className={[
        "flex min-h-0 flex-col",
        showTransport ? "h-full" : "",
        disabled ? "opacity-60" : "",
      ].join(" ")}
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="text-sm font-semibold tracking-tight text-neutral-900 dark:text-neutral-50">
            Selected race
          </div>
          <div className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">
            Drag to choose a round
          </div>
        </div>

        <div className="shrink-0 flex items-center gap-2">
          {raceLoading ? (
            <Pill>
              <Loader2 size={14} className="animate-spin" />
              Loading…
            </Pill>
          ) : raceReady ? (
            <Pill>
              <Check size={14} />
              Ready
            </Pill>
          ) : (
            <Pill>Waiting…</Pill>
          )}
        </div>
      </div>

      {/* Wheel */}
      <div
        className={[
          "mt-4 flex min-h-0 flex-col",
          showTransport ? "flex-1" : "",
        ].join(" ")}
      >
        <div
          ref={surfaceRef}
          className={[
            "relative select-none overflow-hidden rounded-2xl",
            "bg-white dark:bg-neutral-950/40",
            "ring-1 ring-black/5 dark:ring-white/10",
            disabled ? "cursor-not-allowed" : "cursor-grab active:cursor-grabbing",
          ].join(" ")}
          style={{ height: VISIBLE * ITEM_H }}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          onLostPointerCapture={onLostPointerCapture}
        >
          {/* Center highlight band */}
          <div
            className="pointer-events-none absolute left-2 right-2 rounded-xl"
            style={{
              top: SELECT_SLOT * ITEM_H,
              height: ITEM_H,
              background: "rgb(var(--accent))",
              boxShadow: "none",
            }}
          />

          <div
            className="will-change-transform"
            style={{
              transform: `translateY(${padTop - offsetPx}px)`,
              transition:
                isDraggingRef.current || suppressAnimRef.current
                  ? "none"
                  : "transform 160ms cubic-bezier(.2,.8,.2,1)",
            }}
          >
            {loopList.map((_, i) => {
              const baseIdx = normalizeIndex(i);
              const row = baseList[baseIdx];

              const id = row?.id ?? "";
              const name = row?.name ?? "—";
              const active = String(id) === selectedId;

              return (
                <div
                  key={`${i}-${String(id)}`}
                  className={[
                    "flex h-[44px] w-full items-center justify-between px-4",
                    "text-sm",
                    active
                      ? "text-[rgb(var(--accent-fg))] font-semibold"
                      : "text-neutral-500 dark:text-neutral-300",
                  ].join(" ")}
                  role="button"
                  tabIndex={disabled ? -1 : 0}
                  onClick={() => {
                    if (disabled) return;
                    if (movedRef.current) return;

                    const snapped = normalizeOffset(i * ITEM_H);
                    setOffsetPx(snapped);
                    snapToNearest(snapped);

                    if (!active) emit(id);
                  }}
                  onKeyDown={(e) => {
                    if (disabled) return;
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      const snapped = normalizeOffset(i * ITEM_H);
                      setOffsetPx(snapped);
                      snapToNearest(snapped);

                      if (!active) emit(id);
                    }
                  }}
                >
                  <span className="w-10 text-right tabular-nums">
                    {String(baseIdx + 1).padStart(2, "0")}
                  </span>

                  <span className="ml-3 flex-1 truncate">{name}</span>

                  {active ? (
                    <span
                      className="ml-3 inline-flex items-center"
                      style={{ color: "rgb(var(--accent))" }}
                      aria-label="Selected"
                      title="Selected"
                    >
                      <Check size={16} />
                    </span>
                  ) : (
                    <span className="ml-3 w-4" />
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {showYearSelector ? (
          <div className="mt-3">
            <YearStepper
              value={selectedYear}
              options={years}
              onChange={onYearChange}
              disabled={disabled || yearDisabled}
            />
          </div>
        ) : null}

        {/* Transport controls (only when dashboard mode) */}
        {showTransport ? (
          <div className="mt-4 flex items-center justify-center gap-4">
            <IconButton
              disabled={!canStart && !canResume}
              title={raceEverStarted ? "Resume" : "Start"}
              active={playPrimary}
              accentIcon
              onClick={() => {
                if (raceEverStarted) onResume?.();
                else onStart?.();
              }}
            >
              <Play size={18} />
            </IconButton>

            <IconButton
              disabled={!canPause}
              title="Pause"
              active={pausePrimary}
              onClick={() => onPause?.()}
            >
              <Pause size={18} />
            </IconButton>

            <SpeedPill
              value={speed}
              onChange={(s) => onSpeedChange?.(s)}
              disabled={!canTransport}
            />
          </div>
        ) : null}
      </div>
    </div>
  );
}