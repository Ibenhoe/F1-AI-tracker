import { cn } from "../../lib/cn.js";

export default function Card({ className, children, clip = false }) {
  return (
    <div
      className={cn(
        "relative rounded-2xl",
        clip ? "overflow-hidden" : "overflow-visible",

        "bg-white ring-1 ring-neutral-200/70",
        "shadow-sm",

        "dark:bg-neutral-950/40 dark:ring-white/10",
        "dark:shadow-none",

        className
      )}
    >
      {children}
    </div>
  );
}