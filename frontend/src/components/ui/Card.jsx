import { cn } from "../../lib/cn.js";

export default function Card({
  className,
  children,
  clip = false,
  hover = false,
  bordered = false,
}) {
  return (
    <div
      className={cn(
        "relative rounded-2xl transition-shadow duration-200",

        // overflow handling
        clip ? "overflow-hidden" : "overflow-visible",

        // light mode: iOS-like surface
        "bg-white",
        bordered ? "ring-1 ring-neutral-200/70" : "ring-0",
        "shadow-[0_1px_2px_rgba(0,0,0,0.04),0_8px_24px_rgba(0,0,0,0.06)]",

        // dark mode: pure black surface
        "dark:bg-[rgb(var(--panel))]",
        bordered ? "dark:ring-1 dark:ring-white/10" : "dark:ring-0",
        "dark:shadow-none",

        // subtle hover elevation (optional)
        hover &&
  "hover:shadow-[0_2px_6px_rgba(0,0,0,0.06),0_16px_40px_rgba(0,0,0,0.10)]",

        className
      )}
    >
      {children}
    </div>
  );
}