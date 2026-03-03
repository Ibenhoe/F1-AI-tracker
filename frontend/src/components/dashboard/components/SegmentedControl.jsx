// src/components/dashboard/components/SegmentedControl.jsx
export default function SegmentedControl({ value, onChange, items, ariaLabel }) {
  // Guard: avoid division-by-zero + undefined items
  if (!Array.isArray(items) || items.length === 0) return null;

  const activeIndex = Math.max(0, items.findIndex((i) => i.id === value));

  return (
    <div
      className={[
        "relative inline-flex w-full items-stretch justify-center",
        "rounded-2xl p-1",
        "bg-transparent",
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
            onClick={() => onChange?.(item.id)}
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