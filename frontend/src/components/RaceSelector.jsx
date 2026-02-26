import { useEffect, useMemo, useRef, useState } from "react";
import { Loader2, Check, Play, Pause } from "lucide-react";

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
  22: "Abu Dhabi",
};

const ITEM_H = 44;
const VISIBLE = 5;
const MID = Math.floor(VISIBLE / 2);
const LOOPS = 3;

function Pill({ children }) {
  return (
    <span className="inline-flex items-center gap-2 rounded-full border border-neutral-200/60 bg-white/50 px-3 py-1 text-xs text-neutral-700 backdrop-blur dark:border-white/10 dark:bg-neutral-950/30 dark:text-neutral-200">
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
        "transition-all duration-150",
        "active:scale-[0.98]",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[rgb(var(--accent))] focus-visible:ring-offset-2",
        "focus-visible:ring-offset-white dark:focus-visible:ring-offset-neutral-950",
        disabled ? "opacity-40 cursor-not-allowed" : "cursor-pointer",
        active
          ? "bg-[rgb(var(--accent))] text-[rgb(var(--accent-fg))]"
          : "bg-neutral-900/5 hover:bg-neutral-900/10 dark:bg-white/10 dark:hover:bg-white/15",
        "ring-1 ring-neutral-200/70 dark:ring-white/10",
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
        "bg-white/60 backdrop-blur",
        "ring-1 ring-neutral-200/70",
        "dark:bg-neutral-950/30 dark:ring-white/10",
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
                : "text-neutral-600 hover:bg-neutral-100/70 dark:text-neutral-300 dark:hover:bg-white/5",
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

export default function RaceSelector({
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
}) {
  const baseList = useMemo(
    () => Object.entries(RACES).map(([id, name]) => ({ id: Number(id), name })),
    []
  );

  const N = baseList.length;

  const loopList = useMemo(() => {
    const out = [];
    for (let i = 0; i < LOOPS; i++) out.push(...baseList);
    return out;
  }, [baseList]);

  const value = Number(selectedRace ?? 1);

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

  // offset in px
  const [offsetPx, _setOffsetPx] = useState(0);
  const offsetRef = useRef(0);

  const setOffsetPx = (next) => {
    offsetRef.current = next;
    _setOffsetPx(next);
  };

  const padTop = MID * ITEM_H;

  // infinite normalization (no jumps)
  const oneLoopPx = N * ITEM_H;
  const centerLoopStartPx = oneLoopPx;

  const normalizeId = (id) => {
    const raw = Number(id);
    if (!Number.isFinite(raw)) return 1;
    const m = ((raw - 1) % N + N) % N;
    return m + 1;
  };

  const normalizeOffset = (offPx) => {
    let x = offPx - centerLoopStartPx;
    x = ((x % oneLoopPx) + oneLoopPx) % oneLoopPx;
    return centerLoopStartPx + x;
  };

  const clampIndex = (idx) => Math.max(0, Math.min(loopList.length - 1, idx));
  const idxFromOffset = (offPx) => clampIndex(Math.round(offPx / ITEM_H));

  const snapToNearest = (offPx) => {
    const idx = idxFromOffset(offPx);
    const snapped = idx * ITEM_H;
    setOffsetPx(snapped);

    const row = loopList[idx];
    if (!row) return;

    const id = normalizeId(row.id);
    if (id !== value) emit(id);
  };

  // sync from parent to middle loop
  useEffect(() => {
    const baseIdx = baseList.findIndex((r) => r.id === value);
    if (baseIdx < 0) return;

    const target = centerLoopStartPx + baseIdx * ITEM_H;
    setOffsetPx(target);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, baseList]);

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
      } catch {}
    }
    pointerIdRef.current = null;

    // tap-to-select within wheel (no drag)
    if (!disabled && isDraggingRef.current && movedRef.current === false) {
      const surface = surfaceRef.current;
      if (surface) {
        const rect = surface.getBoundingClientRect();
        const y = e.clientY - rect.top;
        const slot = Math.max(0, Math.min(VISIBLE - 1, Math.floor(y / ITEM_H)));
        const deltaSlots = slot - MID;

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

  // ✅ Primary action highlight (action-based, not state-based)
  const playPrimary = canTransport && !raceRunning; // paused/ready -> Play primary
  const pausePrimary = canTransport && raceRunning; // running -> Pause primary

  return (
    <div className={["flex h-full flex-col", disabled ? "opacity-60" : ""].join(" ")}>
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
      <div className="mt-4">
        <div
          ref={surfaceRef}
          className={[
            "relative select-none overflow-hidden rounded-2xl",
            "ring-1 ring-neutral-200/70 bg-white/60 backdrop-blur",
            "dark:ring-white/10 dark:bg-neutral-950/30",
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
            className="pointer-events-none absolute left-2 right-2 top-1/2 -translate-y-1/2 rounded-xl"
            style={{
              height: ITEM_H,
              background: "rgb(var(--accent))",
              boxShadow: "none",
            }}
          />

          {/* Top/bottom fade like iOS pickers */}
          <div className="pointer-events-none absolute inset-x-0 top-0 h-10 bg-gradient-to-b from-white/90 to-transparent dark:from-neutral-950/70" />
          <div className="pointer-events-none absolute inset-x-0 bottom-0 h-10 bg-gradient-to-t from-white/90 to-transparent dark:from-neutral-950/70" />

          <div
            className="will-change-transform"
            style={{
              transform: `translateY(${padTop - offsetPx}px)`,
              transition: isDraggingRef.current
                ? "none"
                : "transform 160ms cubic-bezier(.2,.8,.2,1)",
            }}
          >
            {loopList.map((r, i) => {
              const id = normalizeId(r.id);
              const name = baseList[id - 1]?.name ?? r.name;
              const active = id === value;

              return (
                <div
                  key={`${i}-${id}`}
                  className={[
                    "flex h-[44px] w-full items-center justify-between px-4",
                    "text-sm",
                    active
                      ? "text-[rgb(var(--accent-fg))] font-semibold"
                      : "text-neutral-500 dark:text-neutral-400",
                  ].join(" ")}
                  role="button"
                  tabIndex={disabled ? -1 : 0}
                  onClick={() => {
                    if (disabled) return;
                    if (movedRef.current) return;
                    const snapped = normalizeOffset(i * ITEM_H);
                    setOffsetPx(snapped);
                    snapToNearest(snapped);
                    if (id !== value) emit(id);
                  }}
                  onKeyDown={(e) => {
                    if (disabled) return;
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      const snapped = normalizeOffset(i * ITEM_H);
                      setOffsetPx(snapped);
                      snapToNearest(snapped);
                      if (id !== value) emit(id);
                    }
                  }}
                >
                  <span className="w-10 text-right tabular-nums">
                    {String(id).padStart(2, "0")}
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

        {/* Transport controls */}
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
      </div>
    </div>
  );
}