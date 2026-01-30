import { cn } from "../../lib/cn.js";

export default function Card({ className, children }) {
  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-2xl border border-neutral-800/70",
        "bg-gradient-to-b from-neutral-900/60 to-neutral-950/60",
        "shadow-[0_0_0_1px_rgba(255,255,255,0.02),0_20px_60px_rgba(0,0,0,0.35)]",
        "p-6",
        className
      )}
    >
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(80%_60%_at_50%_0%,rgba(255,255,255,0.06),transparent_60%)]" />
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(90%_80%_at_50%_120%,rgba(0,0,0,0.55),transparent_55%)]" />

      <div className="relative">{children}</div>
    </div>
  );
}
