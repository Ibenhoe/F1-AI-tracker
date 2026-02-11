import { cn } from "../../lib/cn.js";

export default function Card({ className, children, clip = false }) {
  return (
    <div
      className={cn(
        "relative rounded-2xl border",
        clip ? "overflow-hidden" : "overflow-visible",

        // LIGHT
        "border-neutral-200 bg-white",
        "shadow-[0_1px_0_rgba(0,0,0,0.04),0_12px_28px_rgba(0,0,0,0.08)]",

        // DARK (clean + flat)
        "dark:border-neutral-800 dark:bg-neutral-900",
        "dark:shadow-[0_1px_0_rgba(255,255,255,0.04),0_18px_50px_rgba(0,0,0,0.65)]",

        "p-6",
        className
      )}
    >
      {children}
    </div>
  );
}
