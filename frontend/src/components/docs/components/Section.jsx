export default function Section({ id, title, children }) {
  return (
    <section id={id} className="scroll-mt-24">
      <div
        className={[
          "rounded-2xl",
          "bg-white dark:bg-neutral-950/40",
          "ring-1 ring-black/5 dark:ring-white/10",
          "p-6",
          "space-y-5",
        ].join(" ")}
      >
        {/* Header */}
        <div className="space-y-2">
          <h2 className="text-lg font-semibold tracking-tight text-neutral-900 dark:text-neutral-50">
            {title}
          </h2>
          <div className="h-px w-full bg-black/5 dark:bg-white/10" />
        </div>

        {/* Content */}
        <div className="space-y-4 text-sm leading-relaxed text-neutral-700 dark:text-neutral-300">
          {children}
        </div>
      </div>
    </section>
  );
}