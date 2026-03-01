export default function Callout({ label, children }) {
  return (
    <div
      className={[
        "my-5 overflow-hidden rounded-2xl",
        "flex",
        "bg-white dark:bg-neutral-950/40",
        "ring-1 ring-black/5 dark:ring-white/10",
      ].join(" ")}
    >
      {/* Accent rail */}
      <div
        className="w-1 shrink-0"
        style={{ background: "rgb(var(--accent))" }}
        aria-hidden="true"
      />

      {/* Content */}
      <div
        className={[
          "flex-1 p-4",
          "bg-[rgba(var(--accent),0.06)] dark:bg-[rgba(var(--accent),0.10)]",
        ].join(" ")}
      >
        {label ? (
          <p className="text-[11px] font-semibold uppercase tracking-widest text-neutral-600 dark:text-neutral-300">
            {label}
          </p>
        ) : null}

        <div className="mt-2 text-sm leading-relaxed text-neutral-800 dark:text-neutral-200">
          {children}
        </div>
      </div>
    </div>
  );
}