import { cn } from "../../lib/cn.js";

const base =
  "inline-flex items-center rounded-full px-2.5 py-1 text-xs font-medium transition-colors";

const variants = {
  neutral: cn(
    // LIGHT
    "bg-neutral-100 text-neutral-700",
    // DARK
    "dark:bg-white/10 dark:text-neutral-200"
  ),

  accent: cn(
    // LIGHT
    "bg-[rgb(var(--accent)_/_0.12)] text-[rgb(var(--accent))]",
    // DARK
    "dark:bg-[rgb(var(--accent)_/_0.18)] dark:text-[rgb(var(--accent))]"
  ),

  success: cn(
    "bg-emerald-50 text-emerald-700",
    "dark:bg-emerald-950/40 dark:text-emerald-200"
  ),

  warning: cn(
    "bg-amber-50 text-amber-800",
    "dark:bg-amber-950/40 dark:text-amber-200"
  ),

  danger: cn(
    "bg-red-50 text-red-700",
    "dark:bg-red-950/40 dark:text-red-200"
  ),
};

export default function Badge({ className, variant = "neutral", children }) {
  return (
    <span className={cn(base, variants[variant] ?? variants.neutral, className)}>
      {children}
    </span>
  );
}