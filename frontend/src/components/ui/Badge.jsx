import { cn } from "../../lib/cn.js";

const base =
  "inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-medium transition-colors";

const variants = {
  neutral: cn(
    // LIGHT
    "border-neutral-200/80 bg-white/70 text-neutral-700 backdrop-blur",
    // DARK
    "dark:border-white/10 dark:bg-neutral-950/50 dark:text-neutral-300"
  ),

  accent: cn(
    // LIGHT
    "border-[rgb(var(--accent)_/_0.22)] bg-[rgb(var(--accent)_/_0.08)] text-[rgb(var(--accent))]",
    // DARK
    "dark:border-[rgb(var(--accent)_/_0.32)] dark:bg-[rgb(var(--accent)_/_0.14)] dark:text-[rgb(var(--accent))]"
  ),

  success: cn(
    "border-emerald-200/70 bg-emerald-50/70 text-emerald-700",
    "dark:border-emerald-900/50 dark:bg-emerald-950/30 dark:text-emerald-300"
  ),

  warning: cn(
    "border-amber-200/70 bg-amber-50/70 text-amber-800",
    "dark:border-amber-900/50 dark:bg-amber-950/30 dark:text-amber-300"
  ),

  danger: cn(
    "border-red-200/70 bg-red-50/70 text-red-700",
    "dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-300"
  ),
};

export default function Badge({ className, variant = "neutral", children }) {
  return (
    <span
      className={cn(
        base,
        variants[variant] ?? variants.neutral,
        className
      )}
    >
      {children}
    </span>
  );
}
