import { cn } from "../../lib/cn.js";

export default function Card({ className, children, clip = false }) {
  return (
    <div
      className={cn(
        "relative rounded-2xl border",
        clip ? "overflow-hidden" : "overflow-visible",

        "border-neutral-200/80 bg-white",
        "shadow-[0_1px_0_rgba(0,0,0,0.04),0_10px_24px_rgba(0,0,0,0.06)]",

        "dark:border-white/10 dark:bg-neutral-950/60",
        "dark:shadow-[0_1px_0_rgba(255,255,255,0.04),0_18px_50px_rgba(0,0,0,0.55)]",

        "before:pointer-events-none before:absolute before:inset-0 before:rounded-2xl before:content-['']",
        "before:shadow-[inset_0_1px_0_rgba(255,255,255,0.06)] dark:before:shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]",

        "p-6",
        className
      )}
    >
      {children}
    </div>
  );
}
