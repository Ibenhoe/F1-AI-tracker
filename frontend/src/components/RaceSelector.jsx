import { useEffect, useMemo, useRef, useState } from "react";
import { Loader2, Check } from "lucide-react";

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

export default function RaceSelector({
  selectedRace,
  onRaceChange,
  onSelectRace,
  disabled,

  raceLoading = false,
  raceReady = false,
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

  // Debounce timer om geen rapid API calls te doen tijdens scrollen
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
    }, 400);
  };

  // offset in px
  const [offsetPx, _setOffsetPx] = useState(0);
  const offsetRef = useRef(0);

  const [showReadyCheck, setShowReadyCheck] = useState(false);

  const setOffsetPx = (next) => {
    offsetRef.current = next;
    _setOffsetPx(next);
  };

  const padTop = MID * ITEM_H;

  // --- infinite normalization (NO jumps) ---
  const oneLoopPx = N * ITEM_H;
  const centerLoopStartPx = oneLoopPx; // middle copy start

  const normalizeId = (id) => {
    const raw = Number(id);
    if (!Number.isFinite(raw)) return 1;
    const m = ((raw - 1) % N + N) % N;
    return m + 1;
  };

  const normalizeOffset = (offPx) => {
    // Keep offset always inside the middle loop range:
    // [centerLoopStartPx, centerLoopStartPx + oneLoopPx)
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

  // When parent value changes, center to that value in the middle loop
  useEffect(() => {
    const baseIdx = baseList.findIndex((r) => r.id === value);
    if (baseIdx < 0) return;

    const target = centerLoopStartPx + baseIdx * ITEM_H;
    setOffsetPx(target);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, baseList]);

  // Cleanup debounce timer on unmount
  useEffect(() => {
    return () => {
      if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current);
    };
  }, []);

  useEffect(() => {
    if (!raceReady) return;

    setShowReadyCheck(true);

    const t = setTimeout(() => {
      setShowReadyCheck(false);
    }, 2000); // 2 seconds

    return () => clearTimeout(t);
  }, [raceReady]);

  // --- drag state ---
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
    // release capture
    if (pointerIdRef.current !== null) {
      try {
        e.currentTarget.releasePointerCapture?.(pointerIdRef.current);
      } catch { }
    }
    pointerIdRef.current = null;

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

  const onRowClick = (loopIndex, id) => {
    if (disabled) return;
    if (movedRef.current) return; // ignore click after drag

    const snapped = normalizeOffset(loopIndex * ITEM_H);
    setOffsetPx(snapped);
    snapToNearest(snapped);

    if (id !== value) emit(id);
  };

  const showLoading = raceLoading;
  const showReady = showReadyCheck;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-sm font-semibold tracking-tight">Selected race</div>
          <div className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">
            Drag to choose a round
          </div>
        </div>
        <div className="text-xs text-neutral-500 dark:text-neutral-400">
          {disabled ? "Locked while running" : null}
        </div>
      </div>

      <div
        className={[
          "relative select-none rounded-2xl border p-3",
          "border-neutral-200/80 bg-white/80 backdrop-blur",
          "dark:border-white/10 dark:bg-neutral-950/50",
          disabled ? "opacity-60" : "",
        ].join(" ")}
      >
        {/* Center highlight window */}
        <div
          className={[
            "pointer-events-none absolute left-3 right-3 top-1/2 -translate-y-1/2",
            "rounded-xl border",
            "border-neutral-200/70 bg-white/60",
            "dark:border-white/10 dark:bg-white/5",
          ].join(" ")}
          style={{ height: ITEM_H }}
        />

        {/* Fades */}
        <div className="pointer-events-none absolute inset-x-3 top-3 h-10 bg-gradient-to-b from-white/90 to-transparent dark:from-neutral-950/70" />
        <div className="pointer-events-none absolute inset-x-3 bottom-3 h-10 bg-gradient-to-t from-white/90 to-transparent dark:from-neutral-950/70" />

        {/* Drag surface */}
        <div
          ref={surfaceRef}
          className={[
            "relative overflow-hidden rounded-xl",
            disabled ? "cursor-not-allowed" : "cursor-grab active:cursor-grabbing",
          ].join(" ")}
          style={{ height: VISIBLE * ITEM_H }}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          onLostPointerCapture={onLostPointerCapture}
        >
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
                <button
                  key={`${i}-${id}`}
                  type="button"
                  onClick={() => onRowClick(i, id)}
                  className={[
                    "flex h-[44px] w-full items-center justify-between rounded-xl px-3 text-left",
                    "border border-transparent transition-colors",
                    disabled ? "cursor-not-allowed" : "cursor-pointer",
                    active
                      ? [
                        "bg-[rgb(var(--accent))]",
                        "text-[rgb(var(--accent-fg))]",
                        "shadow-[0_0_0_1px_rgb(var(--accent)_/_0.22)_inset]",
                      ].join(" ")
                      : "text-neutral-600 dark:text-neutral-400 hover:bg-neutral-100/60 dark:hover:bg-white/5",
                  ].join(" ")}
                >
                  <div className="flex min-w-0 items-center gap-3">
                    <span className="w-8 text-right tabular-nums opacity-80">
                      {String(id).padStart(2, "0")}
                    </span>
                    <span className="truncate">{name}</span>
                  </div>

                  {/* Status indicator IN de geselecteerde race */}
                  {active ? (
                    <span className="ml-3 inline-flex items-center">
                      {showLoading ? (
                        <Loader2
                          size={16}
                          className="animate-spin opacity-90"
                          style={{ color: "rgb(var(--accent-fg))" }}
                          aria-label="Loading race"
                        />
                      ) : showReady ? (
                        <Check
                          size={16}
                          className="opacity-90"
                          style={{ color: "rgb(var(--accent-fg))" }}
                          aria-label="Race ready"
                        />
                      ) : null}
                    </span>
                  ) : null}
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
