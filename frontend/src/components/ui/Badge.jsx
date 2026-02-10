import { cn } from "../../lib/cn.js";

const variants = {
  neutral: cn(
    // LIGHT THEME
    "border-neutral-300 bg-white text-neutral-700",
    // DARK THEME
    "dark:border-neutral-800 dark:bg-neutral-900/60 dark:text-neutral-300"
  ),
  success: cn(
    // LIGHT THEME
    "border-emerald-300 bg-emerald-50 text-emerald-700",
    // DARK THEME
    "dark:border-emerald-900/60 dark:bg-emerald-950/40 dark:text-emerald-200"
  ),
  warning: cn(
    // LIGHT THEME
    "border-amber-300 bg-amber-50 text-amber-800",
    // DARK THEME
    "dark:border-amber-900/60 dark:bg-amber-950/40 dark:text-amber-200"
  ),
  danger: cn(
    // LIGHT THEME
    "border-red-300 bg-red-50 text-red-700",
    // DARK THEME
    "dark:border-red-900/60 dark:bg-red-950/40 dark:text-red-200"
  ),
};

export default function Badge({ className, variant = "neutral", children }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-medium",
        variants[variant] ?? variants.neutral,
        className
      )}
    >
      {children}
    </span>
  );
}
