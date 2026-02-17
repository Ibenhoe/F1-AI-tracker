import { cn } from "../../lib/cn.js";

const variants = {
  neutral: cn(
    // LIGHT
    "border-neutral-200 bg-white text-neutral-700",
    // DARK
    "dark:border-neutral-800 dark:bg-neutral-900 dark:text-neutral-200"
  ),

  accent: cn(
    // LIGHT
    "border-[rgb(var(--accent)_/_0.25)] bg-[rgb(var(--accent)_/_0.10)] text-[rgb(var(--accent))]",
    // DARK
    "dark:border-[rgb(var(--accent)_/_0.35)] dark:bg-[rgb(var(--accent)_/_0.18)] dark:text-[rgb(var(--accent))]"
  ),

  success: cn(
    "border-emerald-200 bg-emerald-50 text-emerald-700",
    "dark:border-emerald-900/60 dark:bg-emerald-950/40 dark:text-emerald-200"
  ),

  warning: cn(
    "border-amber-200 bg-amber-50 text-amber-800",
    "dark:border-amber-900/60 dark:bg-amber-950/40 dark:text-amber-200"
  ),

  danger: cn(
    "border-red-200 bg-red-50 text-red-700",
    "dark:border-red-900/60 dark:bg-red-950/40 dark:text-red-200"
  ),
};

export default function Badge({ className, variant = "neutral", children }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-medium transition-colors",
        variants[variant] ?? variants.neutral,
        className
      )}
    >
      {children}
    </span>
  );
}
