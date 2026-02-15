import { useEffect, useMemo, useRef, useState } from "react";

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
const VISIBLE = 5; // must be odd
const MID = Math.floor(VISIBLE / 2);
const LOOPS = 3; // render 3x list for infinite feel

export default function RaceSelector({
  selectedRace,
  onRaceChange,
  onSelectRace,
  disabled,
}) {
  const baseList = useMemo(
    () => Object.entries(RACES).map(([id, name]) => ({ id: Number(id), name })),
    []
  );

  const N = baseList.length; // 21
  const loopList = useMemo(() => {
    const out = [];
    for (let i = 0; i < LOOPS; i++) out.push(...baseList);
    return out;
  }, [baseList]);

  const value = Number(selectedRace ?? 1);

  const emit = (id) => {
    if (disabled) return;
    if (onSelectRace) onSelectRace(id);
    else if (onRaceChange) onRaceChange(id);
  };

  // offset is in "rows", but measured in px
  const [offsetPx, _setOffsetPx] = useState(0);
  const offsetRef = useRef(0);

  const setOffsetPx = (next) => {
    offsetRef.current = next;
    _setOffsetPx(next);
  };

  const padTop = MID * ITEM_H;

  // --- infinite wrap helpers ---
  const oneLoopPx = N * ITEM_H; // 21 rows
  const centerLoopStartPx = oneLoopPx; // start of the middle copy (loop 2)

  const normalizeId = (id) => {
    // ensure 1..N
    const raw = Number(id);
    if (!Number.isFinite(raw)) return 1;
    const m = ((raw - 1) % N + N) % N; // 0..N-1
    return m + 1;
  };

  const idxFromOffset = (offPx) => {
    // offPx is the "scroll offset" in px (0 at very top of the whole loopList)
    return Math.round(offPx / ITEM_H);
  };

  const snapToNearest = (offPx) => {
    const idx = idxFromOffset(offPx);
    const snapped = idx * ITEM_H;
    setOffsetPx(snapped);

    const row = loopList[idx];
    if (!row) return;

    const id = normalizeId(row.id);
    if (id !== value) emit(id);
  };

  const wrapIfNeeded = (offPx) => {
    // Keep the offset around the middle loop to avoid hitting edges.
    // Threshold: if we drift into the first or third loop too far, jump by ±oneLoopPx.
    const minSafe = oneLoopPx * 0.5; // halfway through loop 1
    const maxSafe = oneLoopPx * 2.5; // halfway through loop 3

    if (offPx < minSafe) return offPx + oneLoopPx;
    if (offPx > maxSafe) return offPx - oneLoopPx;
    return offPx;
  };

  // When parent value changes, center to that value in the middle loop
  useEffect(() => {
    const baseIdx = baseList.findIndex((r) => r.id === value);
    if (baseIdx < 0) return;

    const target = centerLoopStartPx + baseIdx * ITEM_H;
    setOffsetPx(target);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, baseList]);

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
    // drag down => decrease offset (content moves down)
    let next = startOffsetRef.current - dy;

    // wrap continuously during drag (no clamping)
    next = wrapIfNeeded(next);

    setOffsetPx(next);
  };

  const endDrag = () => {
    if (!isDraggingRef.current) return;
    isDraggingRef.current = false;

    // wrap once more then snap
    const wrapped = wrapIfNeeded(offsetRef.current);
    setOffsetPx(wrapped);
    snapToNearest(wrapped);
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
            {loopList.map((r, i) => {
              const id = normalizeId(r.id);
              const name = baseList[id - 1]?.name ?? r.name;
              const active = id === value;

              return (
                <div
                  key={`${i}-${id}`}
                  className={[
                    "flex items-center justify-between rounded-xl px-3",
                    active
                      ? "text-neutral-950 dark:text-neutral-50"
                      : "text-neutral-500 dark:text-neutral-400",
                  ].join(" ")}
                  style={{ height: ITEM_H }}
                >
                  <div className="flex min-w-0 items-center gap-3">
                    <span className="w-8 text-right tabular-nums">
                      {String(id).padStart(2, "0")}
                    </span>
                    <span className="truncate">{name}</span>
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
