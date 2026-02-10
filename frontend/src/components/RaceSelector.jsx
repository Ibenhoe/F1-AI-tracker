import { useEffect, useMemo, useRef, useState } from "react";

const RACES = {
  1: "Bahrain",
  2: "Saudi Arabia",
  3: "Australia",
  4: "Japan",
  5: "China",
  6: "Miami",
  7: "Monaco",
  8: "Canada",
  9: "Spain",
  10: "Austria",
  11: "United Kingdom",
  12: "Hungary",
  13: "Belgium",
  14: "Netherlands",
  15: "Italy",
  16: "Azerbaijan",
  17: "Singapore",
  18: "Austin",
  19: "Mexico",
  20: "Brazil",
  21: "Abu Dhabi",
};

const ITEM_H = 44; // px per row
const VISIBLE = 5; // must be odd
const MID = Math.floor(VISIBLE / 2);

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}

export default function RaceSelector({
  selectedRace,
  onRaceChange,
  onSelectRace,
  disabled,
}) {
  const list = useMemo(
    () => Object.entries(RACES).map(([id, name]) => ({ id: Number(id), name })),
    []
  );

  const value = Number(selectedRace ?? 21);

  const emit = (id) => {
    if (disabled) return;
    if (onSelectRace) onSelectRace(id);
    else if (onRaceChange) onRaceChange(id);
  };

  const maxIndex = list.length - 1;
  const maxOffset = maxIndex * ITEM_H; // last centered
  const minOffset = 0; // first centered

  // UI state + ref (ref is the truth during dragging)
  const [offsetPx, _setOffsetPx] = useState(0);
  const offsetRef = useRef(0);

  const setOffsetPx = (next) => {
    const clamped = clamp(next, minOffset, maxOffset);
    offsetRef.current = clamped;
    _setOffsetPx(clamped);
  };

  const indexFromOffset = (off) =>
    clamp(Math.round(off / ITEM_H), 0, maxIndex);

  const snapToNearest = (off) => {
    const idx = indexFromOffset(off);
    const snapped = idx * ITEM_H;
    setOffsetPx(snapped);

    const id = list[idx]?.id;
    if (id && id !== value) emit(id);
  };

  // Keep centered when parent value changes
  useEffect(() => {
    const idx = list.findIndex((r) => r.id === value);
    if (idx >= 0) setOffsetPx(idx * ITEM_H);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  // --- drag state ---
  const isDraggingRef = useRef(false);
  const startYRef = useRef(0);
  const startOffsetRef = useRef(0);
  const pointerIdRef = useRef(null);

  const onPointerDown = (e) => {
    if (disabled) return;

    isDraggingRef.current = true;
    pointerIdRef.current = e.pointerId;

    startYRef.current = e.clientY;
    startOffsetRef.current = offsetRef.current;

    e.currentTarget.setPointerCapture?.(e.pointerId);
  };

  const onPointerMove = (e) => {
    if (!isDraggingRef.current) return;

    const dy = e.clientY - startYRef.current;
    // drag down => move list down => decrease offset
    const next = startOffsetRef.current - dy;
    setOffsetPx(next);
  };

  const endDrag = () => {
    if (!isDraggingRef.current) return;
    isDraggingRef.current = false;

    // IMPORTANT: snap using ref (latest), not state
    snapToNearest(offsetRef.current);
  };

  const onPointerUp = (e) => {
    if (pointerIdRef.current !== null) {
      try {
        e.currentTarget.releasePointerCapture?.(pointerIdRef.current);
      } catch {
        // ignore
      }
    }
    pointerIdRef.current = null;
    endDrag();
  };

  const onLostPointerCapture = () => {
    pointerIdRef.current = null;
    endDrag();
  };

  const padTop = MID * ITEM_H;

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
          "relative rounded-2xl border p-3 select-none",
          "border-neutral-200 bg-white/70",
          "dark:border-neutral-800 dark:bg-neutral-950/40",
          disabled ? "opacity-60" : "",
        ].join(" ")}
      >
        {/* center highlight */}
        <div
          className={[
            "pointer-events-none absolute left-3 right-3 top-1/2 -translate-y-1/2",
            "rounded-xl border",
            "border-neutral-200 bg-neutral-50/80",
            "dark:border-neutral-800 dark:bg-neutral-900/40",
          ].join(" ")}
          style={{ height: ITEM_H }}
        />

        {/* fade top/bottom */}
        <div className="pointer-events-none absolute inset-x-3 top-3 h-10 bg-gradient-to-b from-white to-transparent dark:from-neutral-950/60" />
        <div className="pointer-events-none absolute inset-x-3 bottom-3 h-10 bg-gradient-to-t from-white to-transparent dark:from-neutral-950/60" />

        {/* drag surface */}
        <div
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
            {list.map((r) => {
              const active = r.id === value;
              return (
                <div
                  key={r.id}
                  className={[
                    "flex items-center justify-between rounded-xl px-3",
                    active
                      ? "text-neutral-950 dark:text-neutral-50"
                      : "text-neutral-500 dark:text-neutral-400",
                  ].join(" ")}
                  style={{ height: ITEM_H }}
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <span className="w-7 text-right tabular-nums">{r.id}.</span>
                    <span className="truncate">{r.name}</span>
                  </div>
                  {active ? (
                    <span className="text-xs font-medium text-neutral-500 dark:text-neutral-400">
                      Selected
                    </span>
                  ) : null}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
