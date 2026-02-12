import { useEffect, useMemo, useRef, useState } from "react";
import { Check, ChevronDown } from "lucide-react";
import { cn } from "../../lib/cn.js";

export default function AccentSelect({
  value,
  options,
  onChange,
  className,
  buttonClassName,
}) {
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(0);

  const rootRef = useRef(null);
  const buttonRef = useRef(null);

  const selected = useMemo(
    () => options.find((o) => o.id === value) ?? options[0],
    [options, value]
  );

  const isWhite = value === "white";

  useEffect(() => {
    const idx = Math.max(0, options.findIndex((o) => o.id === value));
    setActiveIndex(idx);
  }, [options, value]);

  useEffect(() => {
    if (!open) return;

    const onDown = (e) => {
      if (!rootRef.current) return;
      if (rootRef.current.contains(e.target)) return;
      setOpen(false);
    };

    document.addEventListener("mousedown", onDown);
    document.addEventListener("touchstart", onDown);

    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("touchstart", onDown);
    };
  }, [open]);

  const commit = (idx) => {
    const opt = options[idx];
    if (!opt) return;
    onChange?.(opt.id);
    setOpen(false);
    buttonRef.current?.focus();
  };

  const onKeyDown = (e) => {
    if (!open) {
      if (e.key === "Enter" || e.key === " " || e.key === "ArrowDown") {
        e.preventDefault();
        setOpen(true);
      }
      return;
    }

    if (e.key === "Escape") {
      e.preventDefault();
      setOpen(false);
      buttonRef.current?.focus();
      return;
    }

    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIndex((i) => Math.min(options.length - 1, i + 1));
      return;
    }

    if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIndex((i) => Math.max(0, i - 1));
      return;
    }

    if (e.key === "Enter") {
      e.preventDefault();
      commit(activeIndex);
    }
  };

  return (
    <div ref={rootRef} className={cn("relative", className)}>
      <button
        ref={buttonRef}
        type="button"
        onClick={() => setOpen((v) => !v)}
        onKeyDown={onKeyDown}
        aria-haspopup="listbox"
        aria-expanded={open}
        className={cn(
          "inline-flex h-10 items-center justify-between gap-2 rounded-lg border px-3 text-sm font-medium",
          // LIGHT
          "border-neutral-200 bg-white text-neutral-900",
          "hover:border-neutral-300",
          "focus:outline-none focus:ring-2 focus:ring-neutral-300/60",
          // DARK
          "dark:border-neutral-800 dark:bg-neutral-950 dark:text-neutral-100",
          "dark:hover:border-neutral-700",
          "dark:focus:ring-neutral-700/60",
          buttonClassName
        )}
      >
        <span className="flex items-center gap-2">
          <span
            className={cn(
              "h-2.5 w-2.5 rounded-full",
              isWhite ? "ring-1 ring-neutral-300 dark:ring-white/25" : ""
            )}
            style={{ backgroundColor: "rgb(var(--accent))" }}
            aria-hidden="true"
          />
          <span className="tabular-nums">{selected?.label ?? "Accent"}</span>
        </span>

        <ChevronDown
          size={16}
          className={cn("opacity-80 transition-transform", open && "rotate-180")}
        />
      </button>

      {open ? (
        <div
          role="listbox"
          tabIndex={-1}
          onKeyDown={onKeyDown}
          className={cn(
            "absolute right-0 z-50 mt-2 w-48 overflow-hidden rounded-xl border shadow-lg",
            // LIGHT
            "border-neutral-200 bg-white text-neutral-900",
            // DARK
            "dark:border-neutral-800 dark:bg-neutral-950 dark:text-neutral-100"
          )}
        >
          {options.map((o, idx) => {
            const isSelected = o.id === value;
            const isActive = idx === activeIndex;
            const optIsWhite = o.id === "white";

            const selectedStyle =
              isSelected && !optIsWhite
                ? { backgroundColor: "rgb(var(--accent) / 0.14)" }
                : undefined;

            return (
              <button
                key={o.id}
                type="button"
                role="option"
                aria-selected={isSelected}
                onMouseEnter={() => setActiveIndex(idx)}
                onClick={() => commit(idx)}
                className={cn(
                  "flex w-full items-center justify-between px-3 py-2 text-left text-sm",
                  "transition-colors",
                  isActive
                    ? "bg-neutral-50 dark:bg-neutral-900"
                    : "bg-transparent",
                  "hover:bg-neutral-50 dark:hover:bg-neutral-900",
                  isSelected && optIsWhite ? "bg-neutral-100 dark:bg-neutral-900" : ""
                )}
                style={selectedStyle}
              >
                <span className="flex items-center gap-2">
                  <span
                    className={cn(
                      "h-2 w-2 rounded-full",
                      optIsWhite ? "ring-1 ring-neutral-300 dark:ring-white/25" : ""
                    )}
                    style={optIsWhite ? { backgroundColor: "#fff" } : { backgroundColor: o.hex }}
                    aria-hidden="true"
                  />
                  <span>{o.label}</span>
                </span>

                {isSelected ? <Check size={16} /> : null}
              </button>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}
