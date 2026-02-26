import { useEffect, useMemo, useRef, useState } from "react";
import { Check, ChevronDown } from "lucide-react";
import { cn } from "../../lib/cn.js";

function AccentDot({ primary, secondary, className }) {
  const hasSecondary = !!secondary && secondary !== primary;

  if (!hasSecondary) {
    return (
      <span
        className={cn(
          "h-2.5 w-2.5 rounded-full ring-1 ring-black/10 dark:ring-white/15",
          className
        )}
        style={{ backgroundColor: primary }}
        aria-hidden="true"
      />
    );
  }

  return (
    <span
      className={cn(
        "h-2.5 w-2.5 rounded-full ring-1 ring-black/10 dark:ring-white/15",
        className
      )}
      style={{
        backgroundImage: `conic-gradient(from 45deg, ${primary} 0deg 180deg, ${secondary} 180deg 360deg)`,
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
          "inline-flex h-10 items-center justify-between gap-2 rounded-2xl px-3 text-sm font-medium",
          "bg-neutral-100 text-neutral-900 hover:bg-neutral-200/70",
          "dark:bg-white/10 dark:text-neutral-100 dark:hover:bg-white/15",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[rgb(var(--accent))]",
          buttonClassName
        )}
      >
        <span className="flex items-center gap-2">
          <AccentDot
            primary="rgb(var(--accent))"
            secondary="rgb(var(--accent-secondary))"
          />
          <span className="tabular-nums">{selected?.label ?? "Accent"}</span>
        </span>

        <ChevronDown size={16} className={cn("opacity-80 transition-transform", open && "rotate-180")} />
      </button>

      {open ? (
        <div
          role="listbox"
          tabIndex={-1}
          onKeyDown={onKeyDown}
          className={cn(
            "absolute right-0 z-50 mt-2 w-56 overflow-hidden rounded-2xl",
            // frosted popover
            "bg-white/80 backdrop-blur-md ring-1 ring-neutral-200/70 shadow-[0_18px_60px_rgba(0,0,0,0.18)]",
            "dark:bg-neutral-950/70 dark:ring-white/10 dark:shadow-[0_22px_70px_rgba(0,0,0,0.55)]"
          )}
        >
          <div className="divide-y divide-neutral-200/70 dark:divide-white/10">
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
                    isActive ? "bg-black/5 dark:bg-white/10" : "bg-transparent",
                    "hover:bg-black/5 dark:hover:bg-white/10"
                  )}
                >
                  <span className="flex items-center gap-2">
                    <AccentDot primary={primary} secondary={secondary} />
                    <span className={cn(isSelected && "font-medium")}>{o.label}</span>
                  </span>

                  {isSelected ? (
                    <Check size={16} className="text-[rgb(var(--accent))]" />
                  ) : null}
                </button>
              );
            })}
          </div>
        </div>
      ) : null}
    </div>
  );
}