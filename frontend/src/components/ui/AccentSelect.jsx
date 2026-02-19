import { useEffect, useMemo, useRef, useState } from "react";
import { Check, ChevronDown } from "lucide-react";
import { cn } from "../../lib/cn.js";

function AccentDot({ primary, secondary, className }) {
  const hasSecondary = !!secondary && secondary !== primary;

  if (!hasSecondary) {
    return (
      <span
        className={cn(
          "h-2.5 w-2.5 rounded-full ring-1 ring-neutral-900/10 dark:ring-white/15",
          className
        )}
        style={{ backgroundColor: primary }}
        aria-hidden="true"
      />
    );
  }

  // Diagonal split (45°)
  return (
    <span
      className={cn(
        "h-2.5 w-2.5 rounded-full ring-1 ring-neutral-900/10 dark:ring-white/15",
        className
      )}
      style={{
        backgroundImage: `
          conic-gradient(
            from 45deg,
            ${primary} 0deg 180deg,
            ${secondary} 180deg 360deg
          )
        `,
      }}
      aria-hidden="true"
    />
  );
}

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

  useEffect(() => {
    const idx = Math.max(0, options.findIndex((o) => o.id === value));
    setActiveIndex(idx);
  }, [options, value]);

  // Click outside to close
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
          // Light mode (white) + Dark mode (near-black)
          "border-neutral-200 bg-white text-neutral-900",
          "dark:border-neutral-800 dark:bg-neutral-950 dark:text-neutral-100",
          "hover:border-neutral-300 dark:hover:border-neutral-700",
          "focus:outline-none focus:ring-2 focus:ring-neutral-300/70 dark:focus:ring-neutral-700/60",
          buttonClassName
        )}
      >
        <span className="flex items-center gap-2">
          {/* Primary + optional secondary dot (minimal, no extra labels) */}
          <AccentDot
            primary="rgb(var(--accent))"
            secondary="rgb(var(--accent-secondary))"
          />

          <span className="tabular-nums">{selected?.label ?? "Accent"}</span>
        </span>

        <ChevronDown size={16} className={cn("opacity-80", open && "rotate-180")} />
      </button>

      {open ? (
        <div
          role="listbox"
          tabIndex={-1}
          onKeyDown={onKeyDown}
          className={cn(
            "absolute right-0 z-50 mt-2 w-56 overflow-hidden rounded-xl border shadow-lg",
            // Light mode (white) + Dark mode (near-black)
            "border-neutral-200 bg-white text-neutral-900",
            "dark:border-neutral-800 dark:bg-neutral-950 dark:text-neutral-100"
          )}
        >
          {options.map((o, idx) => {
            const isSelected = o.id === value;
            const isActive = idx === activeIndex;

            const primary = o.primary;
            const secondary = o.secondary ?? o.primary;

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
                  // Hover/active surfaces should be subtle and mode-aware
                  isActive
                    ? "bg-neutral-100 dark:bg-neutral-900"
                    : "bg-transparent",
                  "hover:bg-neutral-100 dark:hover:bg-neutral-900"
                )}
                style={
                  isSelected
                    ? {
                      backgroundColor: "rgb(var(--accent) / 0.14)",
                    }
                    : undefined
                }
              >
                <span className="flex items-center gap-2">
                  <AccentDot primary={primary} secondary={secondary} />
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
