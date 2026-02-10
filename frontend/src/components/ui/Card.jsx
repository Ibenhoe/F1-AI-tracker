import { cn } from "../../lib/cn.js";

export default function Card({ className, children, clip = false }) {
  return (
    <div
      className={cn(
        "relative rounded-2xl border",
        clip ? "overflow-hidden" : "overflow-visible",

        // LIGHT MODE
        "border-neutral-200 bg-white",
        "shadow-[0_1px_0_rgba(0,0,0,0.03),0_10px_30px_rgba(0,0,0,0.06)]",

        // DARK MODE
        "dark:border-neutral-800/70 dark:bg-gradient-to-b dark:from-neutral-900/60 dark:to-neutral-950/60",
        "dark:shadow-[0_0_0_1px_rgba(255,255,255,0.02),0_20px_60px_rgba(0,0,0,0.35)]",

        "p-6",
        className
      )}
    >
      {/* DARK MODE HIGHLIGHTS */}
      <div className="pointer-events-none absolute inset-0 hidden dark:block bg-[radial-gradient(80%_60%_at_50%_0%,rgba(255,255,255,0.06),transparent_60%)]" />
      <div className="pointer-events-none absolute inset-0 hidden dark:block bg-[radial-gradient(90%_80%_at_50%_120%,rgba(0,0,0,0.55),transparent_55%)]" />

      <div className="relative">{children}</div>
    </div>
  );
}
