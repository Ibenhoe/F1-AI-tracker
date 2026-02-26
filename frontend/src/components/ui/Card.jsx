import { cn } from "../../lib/cn.js";

export default function Card({
  className,
  children,
  clip = false,
  hover = false,
  bordered = true,
}) {
  return (
    <div
      className={cn(
        "relative min-w-0 rounded-2xl",
        hover ? "transition-colors duration-200" : "",

        // overflow handling
        clip ? "overflow-hidden" : "overflow-visible",

        // surface (no shadows)
        "bg-white dark:bg-neutral-950/40",

        // border/ring
        bordered ? "ring-1 ring-black/5 dark:ring-white/10" : "ring-0",

        // hover: slightly stronger border + tiny bg change
        hover &&
          (bordered
            ? "hover:ring-black/10 dark:hover:ring-white/15 hover:bg-black/[0.02] dark:hover:bg-white/[0.03]"
            : "hover:bg-black/[0.02] dark:hover:bg-white/[0.03]"),

        className
      )}
    >
      {children}
    </div>
  );
}